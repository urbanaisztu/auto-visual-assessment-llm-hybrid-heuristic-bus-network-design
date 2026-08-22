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

        tourist_map = env.tourist_map
        distance_map = env.distance_map
        PI_map = env.PI_map
        NI_map = env.NI_map

        # Precompute node demand status and edge visual scores for O(1) lookup
        node_demand = {u: env.G.nodes[u].get('tourist', 0) for u in env.G.nodes()}
        edge_visual_score = {}
        for u, v in env.G.edges():
            length = env.G.edges[u, v].get('length', 1.0)
            pi = env.G.edges[u, v].get('PI', 0.0)
            ni = env.G.edges[u, v].get('NI', 0.0)
            score = (pi - ni) / (length + 1e-6)
            edge_visual_score[(u, v)] = score
            edge_visual_score[(v, u)] = score

        def is_weak_link(u, v):
            u_int, v_int = int(u), int(v)
            if not env.G.has_edge(u_int, v_int):
                return False
            score = edge_visual_score.get((u_int, v_int), 0.0)
            demand_u = node_demand.get(u_int, 0)
            demand_v = node_demand.get(v_int, 0)
            return score <= 0.1 and demand_u == 0 and demand_v == 0

        def get_weak_endpoints(route):
            weak_endpoints = set()
            # Always include depot anchors (start/end)
            weak_endpoints.add(0)
            weak_endpoints.add(len(route)-1)
            # Identify interior nodes adjacent to weak edges
            for i in range(1, len(route)-1):
                u, v, w = route[i-1], route[i], route[i+1]
                if is_weak_link(u, v) or is_weak_link(v, w):
                    weak_endpoints.add(i)
            return sorted(weak_endpoints)

        def repair_path_segment(start_node, end_node, max_steps=500):
            if start_node == end_node:
                return []
            try:
                def weight_func(u_, v_, d):
                    u_int, v_int = int(u_), int(v_)
                    length = d.get('length', 1.0)
                    pi = d.get('PI', 0.0)
                    ni = d.get('NI', 0.0)
                    visual_factor = (pi - ni) / (length + 1e-6)
                    demand_u = node_demand.get(u_int, 0)
                    demand_v = node_demand.get(v_int, 0)
                    demand_score = demand_u + demand_v
                    flow_val = env.flow.get(u_int, {}).get(v_int, 0.0)
                    # Strong bias toward demand-covered paths: zero-demand edges heavily penalized
                    base_weight = length / (1e-4 + max(1e-6, visual_factor) + 0.1 * demand_score + 1e-6)
                    if demand_score == 0:
                        base_weight += 10.0
                    return base_weight + flow_val * 0.2

                path = nx.shortest_path(env.G, source=start_node, target=end_node,
                                      weight=weight_func, method='dijkstra')
                return path[1:]
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, nx.NetworkXError):
                # Fallback: use distance only, with step limit via BFS for safety
                try:
                    path = nx.shortest_path(env.G, source=start_node, target=end_node,
                                          weight='length', method='dijkstra')
                    return path[1:]
                except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, nx.NetworkXError):
                    # Ultimate fallback: greedy neighbor expansion with step cap
                    visited = set([start_node])
                    queue = [(start_node, [])]
                    steps = 0
                    while queue and steps < max_steps:
                        curr, path_so_far = queue.pop(0)
                        steps += 1
                        if curr == end_node:
                            return path_so_far
                        for nbr in list(env.G.neighbors(curr))[:10]:  # limit branching
                            if nbr not in visited:
                                visited.add(nbr)
                                new_path = path_so_far + [nbr]
                                if nbr == end_node:
                                    return new_path
                                if len(new_path) < 20:  # prevent explosion
                                    queue.append((nbr, new_path))
                    return [end_node]

        def compute_demand_coverage(route):
            covered = 0
            for node in route:
                if node_demand.get(int(node), 0) == 1:
                    covered += 1
            return covered

        def compute_route_length(route):
            total = 0.0
            for i in range(1, len(route)):
                u, v = int(route[i-1]), int(route[i])
                if env.G.has_edge(u, v):
                    total += env.G.edges[u, v].get('length', 1.0)
                else:
                    seg = repair_path_segment(u, v, max_steps=200)
                    for j in range(len(seg)):
                        uu = u if j == 0 else seg[j-1]
                        vv = seg[j] if j < len(seg) else v
                        if env.G.has_edge(uu, vv):
                            total += env.G.edges[uu, vv].get('length', 1.0)
            return total

        def repair_full_route(route, start_node, end_node, max_nodes=1000):
            if len(route) < 2:
                return [start_node, end_node] if start_node != end_node else [start_node]
            repaired = [start_node]
            visited = {start_node}
            for i in range(1, len(route)):
                if len(repaired) >= max_nodes:
                    break
                next_node = int(route[i])
                if next_node == repaired[-1]:
                    continue
                if next_node in visited and i != len(route) - 1:
                    continue
                if not env.G.has_edge(repaired[-1], next_node):
                    segment = repair_path_segment(repaired[-1], next_node, max_steps=200)
                    for node in segment:
                        node_int = int(node)
                        if node_int not in visited or node_int == next_node:
                            if len(repaired) >= max_nodes:
                                break
                            repaired.append(node_int)
                            visited.add(node_int)
                else:
                    if next_node not in visited or i == len(route) - 1:
                        repaired.append(next_node)
                        visited.add(next_node)
            # Enforce depot anchoring *before* final validation
            if repaired and repaired[0] != start_node:
                repaired.insert(0, start_node)
            if repaired and repaired[-1] != end_node:
                repaired.append(end_node)
            return repaired

        for i in range(num_routes):
            route1 = [int(n) for n in ind1[i]]
            route2 = [int(n) for n in ind2[i]]

            if len(route1) < 3 or len(route2) < 3:
                continue

            start1, end1 = route1[0], route1[-1]
            start2, end2 = route2[0], route2[-1]

            # Block-aware splicing at weak-link endpoints — prioritize those positions
            endpoints1 = get_weak_endpoints(route1)
            endpoints2 = get_weak_endpoints(route2)

            # Prefer cuts that preserve depot anchoring: only interior points between depots
            valid_cuts1 = [p for p in endpoints1 if 1 <= p <= len(route1)-2]
            valid_cuts2 = [p for p in endpoints2 if 1 <= p <= len(route2)-2]

            # If no weak interior endpoints, fall back to midpoints of longest contiguous non-weak segments
            if not valid_cuts1:
                blocks = []
                current_block = [route1[0]]
                for j in range(1, len(route1)):
                    u, v = route1[j-1], route1[j]
                    if not is_weak_link(u, v):
                        current_block.append(v)
                    else:
                        if len(current_block) > 0:
                            blocks.append(current_block[:])
                        current_block = [v]
                if len(current_block) > 0:
                    blocks.append(current_block)
                if blocks:
                    longest_block = max(blocks, key=len)
                    if len(longest_block) >= 4:
                        mid = len(longest_block) // 2
                        idx_in_route = route1.index(longest_block[mid])
                        valid_cuts1 = [idx_in_route]
                    else:
                        valid_cuts1 = [len(route1)//2]
                else:
                    valid_cuts1 = [len(route1)//2]
            if not valid_cuts2:
                blocks = []
                current_block = [route2[0]]
                for j in range(1, len(route2)):
                    u, v = route2[j-1], route2[j]
                    if not is_weak_link(u, v):
                        current_block.append(v)
                    else:
                        if len(current_block) > 0:
                            blocks.append(current_block[:])
                        current_block = [v]
                if len(current_block) > 0:
                    blocks.append(current_block)
                if blocks:
                    longest_block = max(blocks, key=len)
                    if len(longest_block) >= 4:
                        mid = len(longest_block) // 2
                        idx_in_route = route2.index(longest_block[mid])
                        valid_cuts2 = [idx_in_route]
                    else:
                        valid_cuts2 = [len(route2)//2]
                else:
                    valid_cuts2 = [len(route2)//2]

            # Select cut points — deterministic preference for weak-link endpoints
            cxpoint1 = random.choice(valid_cuts1)
            cxpoint2 = random.choice(valid_cuts2)

            # Enforce minimal viable segment lengths (≥2 including depots)
            cxpoint1 = max(1, min(cxpoint1, len(route1)-2))
            cxpoint2 = max(1, min(cxpoint2, len(route2)-2))

            head1 = route1[:cxpoint1]
            tail1 = route1[cxpoint1:]
            head2 = route2[:cxpoint2]
            tail2 = route2[cxpoint2:]

            new_route1 = head1 + tail2
            new_route2 = head2 + tail1

            # Validate feasibility *before* repair: check if splice preserves depot constraints & minimal connectivity
            if len(head1) < 2 or len(tail2) < 2 or len(head2) < 2 or len(tail1) < 2:
                continue

            # Compute parent demand coverage *before* repair (for fairness)
            orig_demand1 = compute_demand_coverage(route1)
            orig_demand2 = compute_demand_coverage(route2)
            min_orig_demand = min(orig_demand1, orig_demand2)

            # Apply demand-biased shortest-path repair *immediately*, with depot anchoring enforced pre-repair
            try:
                new_route1 = repair_full_route(new_route1, start1, end1)
                new_route2 = repair_full_route(new_route2, start2, end2)
            except Exception:
                continue

            # Post-repair demand check — strict enforcement for Objective 2
            new_demand1 = compute_demand_coverage(new_route1)
            new_demand2 = compute_demand_coverage(new_route2)

            # Reject if demand satisfaction degrades — critical for Objective 2 optimization
            if new_demand1 < min_orig_demand or new_demand2 < min_orig_demand:
                continue

            # Optional lightweight load balancing: reject if imbalance worsens significantly
            # (measured by demand per route deviation from mean)
            all_routes_before = [r for r in ind1 + ind2 if len(r) > 0]
            all_routes_after = [r for r in [new_route1, new_route2] + 
                              [ind1[j] for j in range(len(ind1)) if j != i] +
                              [ind2[j] for j in range(len(ind2)) if j != i] if len(r) > 0]
            if len(all_routes_before) >= 2 and len(all_routes_after) >= 2:
                before_demands = [compute_demand_coverage(r) for r in all_routes_before]
                after_demands = [compute_demand_coverage(r) for r in all_routes_after]
                before_std = np.std(before_demands) if 'np' in globals() else (sum((x - sum(before_demands)/len(before_demands))**2 for x in before_demands) / len(before_demands))**0.5
                after_std = np.std(after_demands) if 'np' in globals() else (sum((x - sum(after_demands)/len(after_demands))**2 for x in after_demands) / len(after_demands))**0.5
                if after_std > before_std * 1.2:
                    continue

            ind1[i] = new_route1
            ind2[i] = new_route2

        # Preserve original number of routes
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