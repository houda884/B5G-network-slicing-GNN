'''
 *
 * Copyright (C) 2020 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * MODIFIED VERSION (SMART-ICT / B5G dataset preparation):
 * adapted from the original UPC datanetAPI.py to read ONLY the
 * "Slicing_sample" dataset (2.2.1-10kprocessed), i.e.:
 *
 *   <data_folder>/
 *       results_slicing_WFQ_1000_routing_<a>_<b>.tar.gz   (simulationResults.txt,
 *                                                          traffic.txt, stability.txt,
 *                                                          input_files.txt, linkUsage.txt)
 *       graphs/graph_<id>.txt        (GML, directed multigraph)
 *       routings/routing_<id>.txt    (matrix of output ports, 1 row per source)
 *       slices/slices_<id>.json      (plain JSON list of slices)
 *
 * Only what is present in that dataset is kept:
 *   - time distribution : EXPONENTIAL only
 *   - size distribution : BINOMIAL only
 *   - routing           : port-matrix files only
 *   - slices            : plain JSON (jsonpickle is NOT needed anymore)
 *   - one flow per (src, dst) pair
 *
 * Any other value raises a DatanetException instead of being silently
 * misread.
'''

# -*- coding: utf-8 -*-

import os
import json
import random
import tarfile
import traceback
from enum import IntEnum

import networkx
import numpy


class DatanetException(Exception):
    """
    Exceptions generated when processing dataset
    """
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class TimeDist(IntEnum):
    """
    Time distributions present in the Slicing dataset (code used in traffic.txt).
    """
    EXPONENTIAL_T = 0


class SizeDist(IntEnum):
    """
    Size distributions present in the Slicing dataset (code used in traffic.txt).
    """
    BINOMIAL_S = 2


class SliceFlow:
    """
    One flow of a slice (entry of "flows" in slices_<id>.json).

    Attributes
    ----------
    origin_node : int
    destination : int
    origin_node_antenna : int
        Antenna the origin node is attached to.
    path : list of int
        Nodes crossed by the flow (origin ... destination).
    bandwidth : int
        Bandwidth of the flow in bps.
    traffic_string : str
        Raw traffic descriptor, same layout as a traffic.txt cell.
    """

    def __init__(self, d):
        self.origin_node = int(d["origin_node"])
        self.destination = int(d["destination"])
        self.origin_node_antenna = int(d["origin_node_antenna"])
        path = d["path"]
        # In the files the path is a string such as "[104, 14, 15, 20]"
        self.path = [int(x) for x in (json.loads(path) if isinstance(path, str) else path)]
        self.bandwidth = int(d["bandwidth"])
        self.traffic_string = d["traffic_string"]

    def __repr__(self):
        return "SliceFlow(%d -> %d, bw=%d)" % (self.origin_node, self.destination, self.bandwidth)


class Slice:
    """
    One network slice.

    Attributes
    ----------
    flows : list of SliceFlow
    delta : float
        Delay requirement of the slice (as given in the dataset).
    type : str
        "eMBB", "URLLC" or "mMTC".
    number : int
        Slice number inside the sample.
    """

    def __init__(self, d):
        self.flows = [SliceFlow(f) for f in d["flows"]]
        self.delta = float(d["delta"])
        self.type = d["type"]
        self.number = int(d["number"])

    def __repr__(self):
        return "Slice(number=%d, type=%s, delta=%s, flows=%d)" % (
            self.number, self.type, self.delta, len(self.flows))


class Sample:
    """
    Results of a single iteration (one simulation) of the dataset.

    Attributes
    ----------
    global_packets, global_losses, global_delay : float
        Network-wide packets transmitted / lost per time unit and average delay.
    maxAvgLambda : float
        Overall traffic intensity of the scenario (first field of traffic.txt).
    simulation_time : float
        First field of stability.txt.
    execution_time : int
        Wall-clock time of the simulation in seconds (``Time(s)`` of stability.txt).
    memory_used : int
        Memory used by the simulation in KB (``Mem(KB)`` of stability.txt).
    performance_matrix : NxN numpy object array
        [i, j] -> {'AggInfo': {...}, 'Flows': [ {...} ]}  (Flows is [] if no traffic i->j)
        Keys: PktsDrop, AvgDelay, AvgLnDelay, p10, p20, p50, p80, p90, Jitter.
    traffic_matrix : NxN numpy object array
        [i, j] -> {'AggInfo': {AvgBw, PktsGen, TotalPktsGen}, 'Flows': [ {...} ]}
        Flow keys: TimeDist, TimeDistParams, AvgBw, PktsGen, TotalPktsGen, ToS,
        SizeDist, SizeDistParams.
    routing_matrix : NxN numpy object array
        [i, j] -> list of nodes of the path from i to j.
    topology_object : networkx.MultiDiGraph
    slices : list of Slice
    port_stats : list (one dict per node) of dict (adjacent node -> link stats)
    """

    def __init__(self):
        self.global_packets = None
        self.global_losses = None
        self.global_delay = None
        self.maxAvgLambda = None
        self.simulation_time = None
        self.execution_time = None
        self.memory_used = None
        self.performance_matrix = None
        self.traffic_matrix = None
        self.routing_matrix = None
        self.topology_object = None
        self.port_stats = None
        self.slices = None

        self.data_set_file = None
        self._results_line = None
        self._traffic_line = None
        self._input_files_line = None
        self._status_line = None
        self._link_usage_line = None
        self._routing_file = None
        self._graph_file = None
        self._slices_file = None

    # ------------------------------------------------------------------ getters
    def get_simulation_time(self):
        return self.simulation_time

    def get_simulation_memory(self):
        return self.memory_used

    def get_execution_time(self):
        return self.execution_time

    def get_global_packets(self):
        return self.global_packets

    def get_global_losses(self):
        return self.global_losses

    def get_global_delay(self):
        return self.global_delay

    def get_maxAvgLambda(self):
        return self.maxAvgLambda

    def get_performance_matrix(self):
        return self.performance_matrix

    def get_srcdst_performance(self, src, dst):
        return self.performance_matrix[src, dst]

    def get_traffic_matrix(self):
        return self.traffic_matrix

    def get_srcdst_traffic(self, src, dst):
        return self.traffic_matrix[src, dst]

    def get_routing_matrix(self):
        return self.routing_matrix

    def get_srcdst_routing(self, src, dst):
        return self.routing_matrix[src, dst]

    def get_active_pairs(self):
        """
        Returns the list of (src, dst) pairs that carry traffic. Handy because
        the matrices are N x N but only a small fraction of pairs are used.
        """
        return [(i, j) for i, j in numpy.argwhere(
            numpy.vectorize(lambda c: len(c['Flows']) > 0, otypes=[bool])(self.traffic_matrix))]

    # ------------------------------------------------------------------- slices
    def get_slices(self):
        return self.slices

    def get_slice(self, id):
        """
        Returns the slice at position id of the slices list.
        NB: Slice.number is NOT unique, it restarts at 0 for each slice type
        (e.g. eMBB 0,1,2 / URLLC 0,1 / mMTC 0,1), so id is a list position.
        """
        return self.slices[id]

    def get_slice_flows(self, id):
        return self.slices[id].flows

    def get_slice_delta(self, id):
        return self.slices[id].delta

    def get_slice_type(self, id):
        return self.slices[id].type

    def get_slice_number(self, id):
        return self.slices[id].number

    # ----------------------------------------------------------------- topology
    def get_topology_object(self):
        return self.topology_object

    def get_network_size(self):
        return self.topology_object.number_of_nodes()

    def get_node_properties(self, id):
        """Dict of node attributes (type, schedulingPolicy, queueSizes, ...) or None."""
        if id in self.topology_object.nodes:
            return self.topology_object.nodes[id]
        return None

    def get_link_properties(self, src, dst):
        """Dict of link attributes (bandwidth, port, ...) or None if no link."""
        if dst in self.topology_object[src]:
            return self.topology_object[src][dst][0]
        return None

    def get_srcdst_link_bandwidth(self, src, dst):
        """Bandwidth in bps of the link src->dst, or -1 if not connected."""
        if dst in self.topology_object[src]:
            return float(self.topology_object[src][dst][0]['bandwidth'])
        return -1

    def get_port_stats(self):
        if self.port_stats is None:
            raise DatanetException("ERROR: The processed dataset doesn't have port performance data")
        return self.port_stats

    # ---------------------------------------------------------------- internals
    def _set_data_set_file_name(self, file):
        self.data_set_file = file

    def _get_data_set_file_name(self):
        return self.data_set_file

    def _set_performance_matrix(self, m):
        self.performance_matrix = m

    def _set_traffic_matrix(self, m):
        self.traffic_matrix = m

    def _set_routing_matrix(self, m):
        self.routing_matrix = m

    def _set_slices(self, s):
        self.slices = s

    def _set_topology_object(self, G):
        self.topology_object = G

    def _set_global_packets(self, x):
        self.global_packets = x

    def _set_global_losses(self, x):
        self.global_losses = x

    def _set_global_delay(self, x):
        self.global_delay = x


class DatanetAPI:
    """
    Iterator over the Slicing dataset. Each iteration yields a Sample.
    """

    # Fields of one result cell (simulationResults.txt), 11 values:
    #   AvgBw(kbps), PktsGen, PktsDrop, AvgDelay, AvgLnDelay, p10, p20, p50, p80, p90, Jitter
    _N_RESULT_FIELDS = 11
    # Cell of a pair without traffic in simulationResults.txt / traffic.txt
    _EMPTY_RESULT_PREFIX = "0,0,0,-1,"
    _EMPTY_TRAFFIC_PREFIX = "-1,"

    def __init__(self, data_folder, intensity_values=[], topology_sizes=[],
                 shuffle=False, seed=None):
        """
        Parameters
        ----------
        data_folder : str
            Folder holding the *.tar.gz files plus the graphs/, routings/ and
            slices/ sub-folders (it can contain them at any depth).
        intensity_values : list of 1 or 2 numbers
            Keep only samples whose traffic intensity (maxAvgLambda) is in
            [min, max]. One value means min = max.
        topology_sizes : list of int
            Keep only samples whose topology has one of these numbers of nodes.
        shuffle : bool
            Shuffle the order of the tar.gz files.
        seed : int
            Seed used when shuffling (default 1234).
        """

        self.data_folder = data_folder
        if len(intensity_values) == 1:
            self.intensity_values = [intensity_values[0], intensity_values[0]]
        else:
            self.intensity_values = intensity_values
        self.topology_sizes = topology_sizes
        self.shuffle = shuffle
        self.seed = seed if seed else 1234

        self._all_tuple_files = []
        self._selected_tuple_files = []
        for root, dirs, files in os.walk(self.data_folder):
            if not all(d in dirs for d in ("graphs", "routings", "slices")):
                continue
            files.sort()
            self._all_tuple_files.extend([(root, f) for f in files if f.endswith("tar.gz")])

    def get_available_files(self):
        """Array of tuples (root directory, filename) found in data_folder."""
        return self._all_tuple_files.copy()

    def set_files_to_process(self, tuple_files_lst):
        """
        Restrict the iteration to a list of tuples taken from get_available_files().
        """
        if not isinstance(tuple_files_lst, list):
            raise DatanetException("ERROR: The argument of set_files_to_process should be a list of tuples -> [(root_dir,file),...]")
        for t in tuple_files_lst:
            if not isinstance(t, tuple) or len(t) != 2:
                raise DatanetException("ERROR: The argument of set_files_to_process should be a list of tuples -> [(root_dir,file),...]")
            if t not in self._all_tuple_files:
                raise DatanetException("ERROR: Selected tupla not belong to the list of tuples returned by get_available_files()")
        self._selected_tuple_files = tuple_files_lst.copy()

    # ------------------------------------------------------------------ routing
    def _read_routing_file(self, routing_file, netSize):
        """
        Reads a routing file: row i = source i, column j = destination j, value =
        output port to use at the *current* node to reach j (-1 = none / arrived).
        Rows end with a trailing comma.
        """
        R = numpy.full((netSize, netSize), -1, dtype=int)
        with open(routing_file, "r") as fd:
            for src, line in enumerate(fd):
                ports = line.strip().split(',')
                if ports and ports[-1] == '':
                    ports.pop()
                if src >= netSize or len(ports) != netSize:
                    raise DatanetException("ERROR: routing file %s doesn't match a %d-node topology" % (routing_file, netSize))
                R[src, :] = [int(p) for p in ports]
        return R

    def _get_routing_port_dst(self, G):
        """node_port_dst[node][port] = next node, built from the 'port' link attribute."""
        node_port_dst = {}
        for node in G:
            node_port_dst[node] = {}
            for destination in G[node].keys():
                node_port_dst[node][G[node][destination][0]['port']] = destination
        return node_port_dst

    def _create_routing_matrix(self, G, routing_file):
        """
        Returns an NxN object array where [i, j] is the list of nodes of the path i -> j.
        """
        if not os.path.isfile(routing_file):
            raise DatanetException("ERROR: routing file %s not found" % routing_file)
        netSize = G.number_of_nodes()
        node_port_dst = self._get_routing_port_dst(G)
        R = self._read_routing_file(routing_file, netSize)
        MatrixPath = numpy.empty((netSize, netSize), dtype=object)
        for src in range(netSize):
            for dst in range(netSize):
                node = src
                path = [node]
                while R[node, dst] != -1:
                    node = node_port_dst[node][R[node, dst]]
                    path.append(node)
                    if len(path) > netSize:
                        raise DatanetException("ERROR: routing loop between %d and %d in %s" % (src, dst, routing_file))
                MatrixPath[src, dst] = path
        return MatrixPath

    # ------------------------------------------------------------------- slices
    def _read_slices_file(self, slices_file):
        """Reads a slices_<id>.json file (one JSON list on a single line) -> list of Slice."""
        with open(slices_file, 'r') as f:
            return [Slice(d) for d in json.load(f)]

    # ----------------------------------------------------------------- iterator
    def __iter__(self):
        """
        Yields
        ------
        s : Sample
            One Sample per valid (status OK) simulation of each selected tar.gz.
        """

        tuple_files = list(self._selected_tuple_files if self._selected_tuple_files
                           else self._all_tuple_files)
        if self.shuffle:
            random.Random(self.seed).shuffle(tuple_files)

        for root, file in tuple_files:
            it = 0
            try:
                tar = tarfile.open(os.path.join(root, file), 'r:gz')
            except Exception:
                traceback.print_exc()
                print("Error opening file: %s" % file)
                continue
            with tar:
                try:
                    dir_name = tar.next().name
                    results_file = tar.extractfile(dir_name + "/simulationResults.txt")
                    traffic_file = tar.extractfile(dir_name + "/traffic.txt")
                    status_file = tar.extractfile(dir_name + "/stability.txt")
                    input_files = tar.extractfile(dir_name + "/input_files.txt")
                    link_usage_file = tar.extractfile(dir_name + "/linkUsage.txt")
                except Exception:
                    traceback.print_exc()
                    print("Error: unexpected content in file: %s" % file)
                    continue

                while True:
                    s = Sample()
                    s._set_data_set_file_name(os.path.join(root, file))

                    # All the files are read in lock-step, one line per simulation
                    s._results_line = results_file.readline().decode().rstrip('\n').rstrip(';')
                    s._traffic_line = traffic_file.readline().decode().rstrip('\n')
                    s._status_line = status_file.readline().decode().rstrip('\n')
                    s._input_files_line = input_files.readline().decode().rstrip('\n')
                    s._link_usage_line = link_usage_file.readline().decode().rstrip('\n')
                    if len(s._results_line) == 0 or len(s._traffic_line) == 0:
                        break

                    if ";OK;" not in s._status_line:
                        continue

                    # Filter traffic intensity
                    if len(self.intensity_values) != 0:
                        specific_intensity = float(s._traffic_line[:s._traffic_line.find('|')])
                        if specific_intensity < self.intensity_values[0] or specific_intensity > self.intensity_values[1]:
                            continue

                    try:
                        # input_files line: <id>;graph_x.txt;routing_x.txt;slices_x.json;
                        used_files = s._input_files_line.split(';')
                        s._graph_file = os.path.join(root, "graphs", used_files[1])
                        s._routing_file = os.path.join(root, "routings", used_files[2])
                        s._slices_file = os.path.join(root, "slices", used_files[3])

                        g = networkx.read_gml(s._graph_file, destringizer=int)

                        # Filter topology size
                        if len(self.topology_sizes) != 0 and len(g) not in self.topology_sizes:
                            continue

                        s._set_topology_object(g)
                        s._set_routing_matrix(self._create_routing_matrix(g, s._routing_file))
                        s._set_slices(self._read_slices_file(s._slices_file))
                        self._process_flow_results(s)
                        self._process_link_usage(s)
                    except (GeneratorExit, SystemExit, KeyboardInterrupt):
                        raise
                    except Exception:
                        traceback.print_exc()
                        print("Error in the file: %s   iteration: %d" % (file, it))
                        it += 1
                        continue

                    it += 1
                    yield s

    # ------------------------------------------------------------ flow results
    def _process_flow_results(self, s):
        """
        Fills global metrics, performance_matrix and traffic_matrix of the Sample
        from its results / traffic / status lines.
        """

        first_params = list(map(float, s._results_line.split('|')[0].split(',')))
        s._set_global_packets(first_params[0])
        s._set_global_losses(first_params[1])
        s._set_global_delay(first_params[2])

        r = s._results_line[s._results_line.find('|') + 1:].split(';')

        ptr = s._traffic_line.find('|')
        t = s._traffic_line[ptr + 1:].rstrip(';').split(';')
        s.maxAvgLambda = float(s._traffic_line[:ptr])

        # stability.txt -> "<sim_time>;<x>;OK;Mem(KB):<mem>;Time(s):<secs>"
        status = s._status_line.split(';')
        sim_time = float(status[0])
        s.simulation_time = sim_time
        s.memory_used = int(status[3].split(":", 1)[1])
        s.execution_time = int(status[4].split(":", 1)[1]) if len(status) > 4 else None

        netSize = s.get_network_size()
        if len(r) != netSize * netSize or len(t) != netSize * netSize:
            raise DatanetException("ERROR: results/traffic have %d/%d cells but topology has %d nodes"
                                   % (len(r), len(t), netSize))

        m_result = numpy.empty((netSize, netSize), dtype=object)
        m_traffic = numpy.empty((netSize, netSize), dtype=object)

        for idx in range(netSize * netSize):
            i, j = divmod(idx, netSize)
            rc, tc = r[idx], t[idx]

            if rc.startswith(self._EMPTY_RESULT_PREFIX) or tc.startswith(self._EMPTY_TRAFFIC_PREFIX):
                # Pair without traffic
                m_result[i, j] = {'AggInfo': self._result_dict(rc.split(',')), 'Flows': []}
                m_traffic[i, j] = {'AggInfo': {'AvgBw': 0.0, 'PktsGen': 0.0, 'TotalPktsGen': 0.0},
                                   'Flows': []}
                continue

            if ':' in rc or ':' in tc:
                raise DatanetException("ERROR: several flows per src-dst pair are not supported by this dataset reader")

            agg = list(map(float, rc.split(',')))
            result = self._result_dict(agg)
            # AvgBw is given in kbps in the file -> bps
            traffic_agg = {'AvgBw': agg[0] * 1000,
                           'PktsGen': agg[1],
                           'TotalPktsGen': agg[1] * sim_time}

            flow = {}
            fields = tc.split(',')
            offset = self._timedistparams(fields, flow)
            flow['AvgBw'] = agg[0] * 1000
            flow['PktsGen'] = agg[1]
            flow['TotalPktsGen'] = sim_time * agg[1]
            flow['ToS'] = int(fields[-1])
            self._sizedistparams(fields, offset, flow)

            m_result[i, j] = {'AggInfo': result, 'Flows': [dict(result)]}
            m_traffic[i, j] = {'AggInfo': traffic_agg, 'Flows': [flow]}

        s._set_performance_matrix(m_result)
        s._set_traffic_matrix(m_traffic)

    def _result_dict(self, v):
        v = list(map(float, v))
        if len(v) != self._N_RESULT_FIELDS:
            raise DatanetException("ERROR: a result cell has %d fields, %d expected" % (len(v), self._N_RESULT_FIELDS))
        return {'PktsDrop': v[2], 'AvgDelay': v[3], 'AvgLnDelay': v[4],
                'p10': v[5], 'p20': v[6], 'p50': v[7], 'p80': v[8], 'p90': v[9],
                'Jitter': v[10]}

    def _timedistparams(self, data, dict_traffic):
        """
        Parses the time distribution of a traffic cell. Only EXPONENTIAL (code 0):
        <0>,<EqLambda>,<AvgPktsLambda>,<ExpMaxFactor>
        Returns the number of fields consumed.
        """
        if data[0] != str(int(TimeDist.EXPONENTIAL_T)):
            raise DatanetException("ERROR: unsupported time distribution '%s' (only EXPONENTIAL_T=0 is present in the Slicing dataset)" % data[0])
        dict_traffic['TimeDist'] = TimeDist.EXPONENTIAL_T
        dict_traffic['TimeDistParams'] = {'EqLambda': float(data[1]),
                                          'AvgPktsLambda': float(data[2]),
                                          'ExpMaxFactor': float(data[3])}
        return 4

    def _sizedistparams(self, data, starting_point, dict_traffic):
        """
        Parses the size distribution of a traffic cell. Only BINOMIAL (code 2):
        <2>,<AvgPktSize>,<PktSize1>,<PktSize2>
        """
        if data[starting_point] != str(int(SizeDist.BINOMIAL_S)):
            raise DatanetException("ERROR: unsupported size distribution '%s' (only BINOMIAL_S=2 is present in the Slicing dataset)" % data[starting_point])
        dict_traffic['SizeDist'] = SizeDist.BINOMIAL_S
        dict_traffic['SizeDistParams'] = {'AvgPktSize': float(data[starting_point + 1]),
                                          'PktSize1': float(data[starting_point + 2]),
                                          'PktSize2': float(data[starting_point + 3])}

    # -------------------------------------------------------------- link usage
    def _process_link_usage(self, s):
        """
        Fills s.port_stats: list (one entry per node) of dict (adjacent node ->
        link statistics + per-QoS-queue statistics).
        """
        l = s._link_usage_line.split(";")
        netSize = s.get_network_size()
        if len(l) < netSize * netSize:
            raise DatanetException("ERROR: linkUsage has %d cells, expected %d" % (len(l), netSize * netSize))

        port_stat = []
        for i in range(netSize):
            # Occupancy is in packets when the node declares queueSizes, bits otherwise
            occupancy_unit = "num_pkts" if "queueSizes" in s.topology_object.nodes[i] else "num_bits"
            port_stat.append({})
            for j in range(netSize):
                cell = l[i * netSize + j]
                if cell == "-1":
                    continue
                params_lst = cell.split(":")
                params = params_lst[0].split(",")
                if params[0] == "-1":
                    continue
                link_stat = {"utilization": float(params[0]),
                             "losses": float(params[1]),
                             "avgPacketSize": float(params[2])}
                qos_queue_stat_lst = []
                for q in params_lst[1:]:
                    qp = q.split(",")
                    qos_queue_stat_lst.append({"utilization": float(qp[0]),
                                               "losses": float(qp[1]),
                                               "avgPortOccupancy": {occupancy_unit: float(qp[2])},
                                               "maxQueueOccupancy": {occupancy_unit: float(qp[3])},
                                               "avgPacketSize": float(qp[4])})
                link_stat["qosQueuesStats"] = qos_queue_stat_lst
                port_stat[i][j] = link_stat

        s.port_stats = port_stat
