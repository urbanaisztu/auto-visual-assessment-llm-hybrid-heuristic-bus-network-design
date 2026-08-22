def cx_operator(input_ind1, input_ind2, env):
    try:
        ind1 = copy.deepcopy(input_ind1)
        ind2 = copy.deepcopy(input_ind2)

        num_routes = min(len(ind1), len(ind2))
        if num_routes == 0:
            if len(ind1) < len(input_ind1):
                ind1.extend(copy.deepcopy(input_ind1[len(ind1):]))
            if len(ind2) < len(input_ind2):
                ind2.extend(copy.deepcopy(input_ind2[len(ind2):]))
            return ind1, ind2

        # Precompute globally once: node demand status and edge visual ratios (bidirectional)
        node_demand = {u: env.G.nodes[u].get('tourist', 0) for u in env.G.nodes()}
        edge_visual_ratio = {}
        for u, v in env.G.edges():
            length = env.G.edges[u, v].get('length', 1.0)
            pi = env.G.edges[u, v].get('PI', 0.0)
            ni = env.G.edges[u, v].get('NI', 0.0)
            vr = (pi - ni) / (length + 1e-6)
            edge_visual_ratio[(u, v)] = vr
            edge_visual_ratio[(v, u)] = vr

        # Demand-aware weak edge detection: only edges with zero demand at both ends AND non-positive visual ratio
        def is_weak_edge(u, v):
            u_int, v_int = int(u), int(v)
            if not env.G.has_edge(u_int, v_int):
                return False
            visual_ratio = edge_visual_ratio.get((u_int, v_int), -1.0)
            demand_u = node_demand.get(u_int, 0)
            demand_v = node_demand.get(v_int, 0)
            return visual_ratio <= 0.0 and demand_u == 0 and demand_v == 0

        def identify_building_blocks(route):
            """Identify cohesive functional blocks: maximal contiguous subpaths where
            every edge is either (a) visually strong (vr > 0.0) OR (b) incident to at least one high-demand node.
            Blocks are indivisible; cuts only allowed *between* blocks."""
            if len(route) < 2:
                return [(0, len(route)-1)]
            
            blocks = []
            start_idx = 0
            i = 1
            while i < len(route):
                u, v = route[i-1], route[i]
                u_int, v_int = int(u), int(v)
                
                visual_ratio = edge_visual_ratio.get((u_int, v_int), -1.0)
                demand_u = node_demand.get(u_int, 0)
                demand_v = node_demand.get(v_int, 0)
                is_strong_or_demand = visual_ratio > 0.0 or demand_u == 1 or demand_v == 1
                
                if is_strong_or_demand:
                    i += 1
                else:
                    if i - 1 > start_idx:
                        blocks.append((start_idx, i - 1))
                    start_idx = i
                    i += 1
            if start_idx < len(route):
                blocks.append((start_idx, len(route)-1))
            return blocks

        def select_cut_points_from_blocks(route, blocks):
            """Select cut points *only* at inter-block boundaries — prioritizing endpoints adjacent to weak links,
            i.e., cut *after* a block ending in a weak-link node (low-demand, low-visual) to enable demand-biased repair."""
            if len(blocks) <= 1:
                return [len(route) // 2]
            
            candidates = []
            for i in range(len(blocks) - 1):
                end_of_block = blocks[i][1]
                cut_idx = end_of_block + 1
                if cut_idx < len(route) - 1:
                    prev_node = route[end_of_block]
                    prev_int = int(prev_node)
                    is_weak_endpoint = node_demand.get(prev_int, 0) == 0
                    if cut_idx < len(route):
                        next_node = route[cut_idx]
                        next_int = int(next_node)
                        if is_weak_edge(prev_int, next_int):
                            candidates.insert(0, cut_idx)  # High-priority: weak link follows
                        else:
                            candidates.append(cut_idx)
                    else:
                        candidates.append(cut_idx)
            
            if not candidates:
                return [len(route) // 2]
            return candidates

        def repair_path_segment(start_node, end_node):
            """Demand-biased shortest path repair: minimize length while maximizing demand coverage and visual quality.
            Uses precomputed maps & bounded Dijkstra with hard limits."""
            start_int, end_int = int(start_node), int(end_node)
            if start_int == end_int:
                return []
            
            try:
                def weight_func(u_, v_, d):
                    u_int, v_int = int(u_), int(v_)
                    length = d.get('length', 1.0)
                    pi = d.get('PI', 0.0)
                    ni = d.get('NI', 0.0)
                    visual_ratio = (pi - ni) / (length + 1e-6)
                    demand_u = node_demand.get(u_int, 0)
                    demand_v = node_demand.get(v_int, 0)
                    demand_score = demand_u + demand_v
                    flow_val = env.flow.get(u_int, {}).get(v_int, 0.0)
                    
                    base_weight = length / (1e-6 + max(visual_ratio, 0.0) + 1e-6)
                    demand_bonus = 0.0
                    if demand_u == 1 or demand_v == 1:
                        demand_bonus = -1.5 * length
                    elif demand_u == 0 and demand_v == 0:
                        demand_bonus = +0.8 * length
                    
                    return base_weight + demand_bonus + 0.002 * flow_val

                # Bounded Dijkstra with explicit iteration cap
                import heapq
                visited = set()
                pq = [(0.0, start_int, [])]
                max_explored = 300
                counter = 0

                while pq and counter < max_explored:
                    cost, node, path = heapq.heappop(pq)
                    if node in visited:
                        continue
                    visited.add(node)
                    counter += 1
                    if node == end_int:
                        return path[1:] + [end_int] if path else [end_int]
                    
                    for neighbor in env.G.neighbors(node):
                        if neighbor in visited:
                            continue
                        try:
                            edge_data = env.G[node][neighbor]
                            w = weight_func(node, neighbor, edge_data)
                            new_cost = cost + w
                            new_path = path + [node]
                            heapq.heappush(pq, (new_cost, neighbor, new_path))
                        except (KeyError, TypeError, ValueError):
                            continue
                
                # Fallback: pure distance shortest path
                path = nx.shortest_path(env.G, source=start_int, target=end_int, weight='length', method='dijkstra')
                return path[1:]
                
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, ValueError, TypeError, MemoryError):
                try:
                    path = nx.shortest_path(env.G, source=start_int, target=end_int, weight='length', method='dijkstra')
                    return path[1:]
                except:
                    return [end_int]

        def compute_demand_coverage(route):
            return sum(1 for node in set(route) if node_demand.get(int(node), 0) == 1)

        for i in range(num_routes):
            route1 = [int(n) for n in ind1[i]]
            route2 = [int(n) for n in ind2[i]]

            if len(route1) < 3 or len(route2) < 3:
                continue

            start1, end1 = route1[0], route1[-1]
            start2, end2 = route2[0], route2[-1]

            # Enforce depot anchoring BEFORE splicing
            if route1 and route1[0] != start1:
                route1 = [start1] + route1
            if route1 and route1[-1] != end1:
                if end1 in route1:
                    route1 = route1[:route1.index(end1)+1]
                else:
                    route1 = route1 + [end1]
            if route2 and route2[0] != start2:
                route2 = [start2] + route2
            if route2 and route2[-1] != end2:
                if end2 in route2:
                    route2 = route2[:route2.index(end2)+1]
                else:
                    route2 = route2 + [end2]

            # Identify building blocks and cut only at inter-block boundaries
            blocks1 = identify_building_blocks(route1)
            blocks2 = identify_building_blocks(route2)
            cuts1 = select_cut_points_from_blocks(route1, blocks1)
            cuts2 = select_cut_points_from_blocks(route2, blocks2)

            cxpoint1 = random.choice(cuts1) if cuts1 else len(route1) // 2
            cxpoint2 = random.choice(cuts2) if cuts2 else len(route2) // 2

            cxpoint1 = max(1, min(cxpoint1, len(route1) - 2))
            cxpoint2 = max(1, min(cxpoint2, len(route2) - 2))

            head1 = route1[:cxpoint1]
            tail1 = route1[cxpoint1:]
            head2 = route2[:cxpoint2]
            tail2 = route2[cxpoint2:]

            new_route1 = head1 + tail2
            new_route2 = head2 + tail1

            # Compute original demand coverage for Pareto-aware acceptance
            orig_demand1, orig_demand2 = compute_demand_coverage(route1), compute_demand_coverage(route2)
            min_demand = min(orig_demand1, orig_demand2)
            max_demand = max(orig_demand1, orig_demand2)

            new_demand1 = compute_demand_coverage(new_route1)
            new_demand2 = compute_demand_coverage(new_route2)
            new_max_demand = max(new_demand1, new_demand2)
            new_min_demand = min(new_demand1, new_demand2)

            # Accept if demand preserved OR imbalance reduced (Pareto improvement on Objective 2)
            accept1 = (new_demand1 >= min_demand) or (new_max_demand < max_demand)
            accept2 = (new_demand2 >= min_demand) or (new_max_demand < max_demand)

            if accept1:
                try:
                    # Repair ONLY the junction between head1 and tail2
                    if len(head1) > 0 and len(tail2) > 0:
                        join_u, join_v = head1[-1], tail2[0]
                        if not env.G.has_edge(join_u, join_v):
                            repaired_junction = repair_path_segment(join_u, join_v)
                            new_route1 = head1 + repaired_junction + tail2[1:]
                        else:
                            new_route1 = head1 + tail2
                    # Enforce depot anchoring AFTER repair
                    if new_route1 and new_route1[0] != start1:
                        new_route1 = [start1] + new_route1
                    if new_route1 and new_route1[-1] != end1:
                        if end1 in new_route1:
                            new_route1 = new_route1[:new_route1.index(end1)+1]
                        else:
                            new_route1 = new_route1 + [end1]
                    ind1[i] = new_route1
                except Exception:
                    pass

            if accept2:
                try:
                    # Repair ONLY the junction between head2 and tail1
                    if len(head2) > 0 and len(tail1) > 0:
                        join_u, join_v = head2[-1], tail1[0]
                        if not env.G.has_edge(join_u, join_v):
                            repaired_junction = repair_path_segment(join_u, join_v)
                            new_route2 = head2 + repaired_junction + tail1[1:]
                        else:
                            new_route2 = head2 + tail1
                    # Enforce depot anchoring AFTER repair
                    if new_route2 and new_route2[0] != start2:
                        new_route2 = [start2] + new_route2
                    if new_route2 and new_route2[-1] != end2:
                        if end2 in new_route2:
                            new_route2 = new_route2[:new_route2.index(end2)+1]
                        else:
                            new_route2 = new_route2 + [end2]
                    ind2[i] = new_route2
                except Exception:
                    pass

        # Maintain original number of routes
        if len(ind1) < len(input_ind1):
            ind1.extend(copy.deepcopy(input_ind1[len(ind1):]))
        elif len(ind1) > len(input_ind1):
            ind1 = ind1[:len(input_ind1)]

        if len(ind2) < len(input_ind2):
            ind2.extend(copy.deepcopy(input_ind2[len(ind2):]))
        elif len(ind2) > len(input_ind2):
            ind2 = ind2[:len(input_ind2)]

        return ind1, ind2

    except Exception as e:
        return copy.deepcopy(input_ind1), copy.deepcopy(input_ind2)