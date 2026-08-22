def mt_operator(input_ind, env):
    try:
        # 1. Deep copy to avoid modifying original
        ind = copy.deepcopy(input_ind)
        if not ind:
            return (ind, )

        # 2. Select a random route
        idx = random.randint(0, len(ind) - 1)
        route = ind[idx]
        if len(route) < 4:
            return (ind, )

        # 3. Precompute per-individual uncovered demand: nodes with env.tourist_map > 0 NOT covered by *any* route in current individual
        all_covered_in_ind = set()
        for r in ind:
            all_covered_in_ind.update(r)
        uncovered_demand_nodes = [u for u in env.tourist_map if env.tourist_map[u] > 0 and u not in all_covered_in_ind]
        uncovered_demand_nodes.sort(key=lambda x: env.tourist_map.get(x, 0), reverse=True)

        # 4. Enforce depot constraints: identify valid depot(s) from route endpoints
        depot_candidates = []
        if len(route) >= 2:
            depot_candidates = [int(route[0]), int(route[-1])]
        depot_candidates = [d for d in depot_candidates if d in env.G.nodes()]

        # 5. Precompute edge coverage frequency across the individual for novelty scoring
        edge_coverage = {}
        for r in ind:
            for i in range(len(r) - 1):
                u, v = int(r[i]), int(r[i + 1])
                key = (u, v)
                edge_coverage[key] = edge_coverage.get(key, 0) + 1

        # 6. Demand-aware distance estimator with caching fallback
        ε = 0.1
        def get_demand_weighted_dist(a, b):
            a_int, b_int = int(a), int(b)
            if a_int == b_int:
                return 0.0
            if not env.G.has_edge(a_int, b_int):
                # Try cost matrix first
                if a_int in env.cost_matrix and b_int in env.cost_matrix[a_int]:
                    base_dist = env.cost_matrix[a_int][b_int]['sum_dis']
                    dem_b = env.tourist_map.get(b_int, 0)
                    return base_dist / (1.0 + ε * dem_b)
                # Fallback to demand-weighted Dijkstra
                try:
                    def weight_func(u_, v_, d_):
                        dem_v = env.tourist_map.get(v_, 0)
                        return d_.get('length', 1.0) / (1.0 + ε * dem_v)
                    path = nx.shortest_path(env.G, a_int, b_int, weight=weight_func, method='dijkstra')
                    dist = sum(env.G.edges[path[i], path[i+1]]['length'] 
                              for i in range(len(path)-1))
                    return dist / (1.0 + ε * env.tourist_map.get(b_int, 0))
                except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, ValueError):
                    return float('inf')
            else:
                dem_b = env.tourist_map.get(b_int, 0)
                return env.G.edges[a_int, b_int]['length'] / (1.0 + ε * dem_b)
            return float('inf')

        # 7. Ghost insertion: targeted on top-k uncovered demand nodes with novelty-aware scoring
        max_ghost_tries = min(3, len(uncovered_demand_nodes))
        candidate_ghosts = uncovered_demand_nodes[:max_ghost_tries]
        best_ghost = None
        best_insert_pos = None
        best_marginal_satisfy = -float('inf')

        route_set = set(route)
        if candidate_ghosts and random.random() < 0.7:
            for candidate_ghost in candidate_ghosts:
                if candidate_ghost not in env.G.nodes():
                    continue
                for i in range(len(route) - 1):
                    u, v = int(route[i]), int(route[i + 1])
                    if u == candidate_ghost or v == candidate_ghost:
                        continue
                    cost_u2g = get_demand_weighted_dist(u, candidate_ghost)
                    cost_g2v = get_demand_weighted_dist(candidate_ghost, v)
                    cost_uv = get_demand_weighted_dist(u, v)
                    if cost_u2g == float('inf') or cost_g2v == float('inf') or cost_uv == float('inf'):
                        continue
                    inc_cost = cost_u2g + cost_g2v - cost_uv
                    sat_score = env.tourist_map.get(candidate_ghost, 0)
                    # Route-local novelty: penalize overused edges (u,v) and (v,u)
                    coverage_uv = edge_coverage.get((u, v), 0)
                    coverage_vu = edge_coverage.get((v, u), 0)
                    novelty = 1.0 / (1e-6 + min(coverage_uv, coverage_vu, 1))
                    score = sat_score * novelty - 0.1 * inc_cost
                    if score > best_marginal_satisfy:
                        best_marginal_satisfy = score
                        best_ghost = candidate_ghost
                        best_insert_pos = i

            if best_ghost is not None and best_insert_pos is not None:
                new_route = (route[:best_insert_pos + 1] +
                             [best_ghost] +
                             route[best_insert_pos + 1:])
                # Repair connectivity
                valid = True
                repaired_route = [new_route[0]]
                for i in range(len(new_route) - 1):
                    a, b = int(new_route[i]), int(new_route[i + 1])
                    if env.G.has_edge(a, b):
                        repaired_route.append(b)
                    else:
                        try:
                            repair_path = nx.shortest_path(env.G, a, b, weight='length')
                            repaired_route.extend(repair_path[1:])
                        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                            valid = False
                            break
                if valid:
                    ind[idx] = repaired_route
                    return (ind, )

        # 8. Ruin-and-recreate: demand-guided local search with depot-aware anchoring
        # Score edges by marginal satisfy gain: 1 if destination node is uncovered high-demand
        edge_satisfy = []
        for i in range(len(route) - 1):
            u, v = int(route[i]), int(route[i + 1])
            sat = float(env.tourist_map.get(v, 0)) if v in uncovered_demand_nodes else 0.0
            edge_satisfy.append(sat)

        # Find worst contiguous segment (lowest avg satisfy) of variable length
        min_avg = float('inf')
        best_window = None
        n = len(edge_satisfy)
        for window_len in range(3, min(9, n + 1)):
            for start in range(n - window_len + 1):
                avg_sat = sum(edge_satisfy[start:start + window_len]) / window_len
                if avg_sat < min_avg:
                    min_avg = avg_sat
                    best_window = (start, start + window_len)

        if best_window is not None:
            start_edge_idx, end_edge_idx = best_window
            u = int(route[start_edge_idx])
            v = int(route[end_edge_idx + 1])
            left_part = route[:start_edge_idx + 1]
            right_part = route[end_edge_idx + 1:]

            visited = set(left_part + right_part)
            path = [u]
            current = u
            remaining_demand = [n for n in uncovered_demand_nodes if n not in visited]
            max_steps = 30
            step = 0

            while current != v and step < max_steps:
                candidates = []
                # Prioritize direct connection to v
                if env.G.has_edge(current, v):
                    candidates = [(v, 0, True)]
                else:
                    # Try neighbors that are uncovered demand nodes first
                    for nbr in env.G.neighbors(current):
                        nbr = int(nbr)
                        if nbr == v:
                            candidates = [(nbr, 0, True)]
                            break
                        if nbr in remaining_demand:
                            candidates.append((nbr, 0, True))
                        elif nbr not in visited and nbr in env.G.nodes():
                            candidates.append((nbr, 1, False))
                
                if not candidates:
                    # Fallback to demand-weighted shortest path
                    try:
                        def demand_weight(a, b, d):
                            dem_b = env.tourist_map.get(b, 0)
                            return d.get('length', 1.0) / (1.0 + ε * dem_b)
                        fallback_path = nx.shortest_path(env.G, current, v, weight=demand_weight, method='dijkstra')
                        if len(fallback_path) > 1:
                            path.extend(fallback_path[1:])
                        else:
                            path.append(v)
                        break
                    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                        try:
                            fallback_path = nx.shortest_path(env.G, current, v, weight='length')
                            path.extend(fallback_path[1:])
                        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                            pass
                        break

                # Sort: prefer demand targets, then non-visited, then others
                candidates.sort(key=lambda x: (x[1], 0 if x[2] else 1))
                next_node = candidates[0][0]
                path.append(next_node)
                visited.add(next_node)
                if next_node in remaining_demand:
                    remaining_demand.remove(next_node)
                current = next_node
                step += 1

            if path and path[-1] == v:
                new_route = left_part[:-1] + path + right_part[1:]
                # Validate and repair edge connectivity
                repaired = True
                for i in range(len(new_route) - 1):
                    if i >= 100:
                        break
                    a, b = int(new_route[i]), int(new_route[i + 1])
                    if not env.G.has_edge(a, b):
                        try:
                            repair_path = nx.shortest_path(env.G, a, b, weight='length')
                            if len(repair_path) > 2:
                                new_route = new_route[:i + 1] + repair_path[1:-1] + new_route[i + 1:]
                        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                            repaired = False
                            break
                if repaired:
                    ind[idx] = new_route
                    return (ind, )

        # 9. Ultimate fallback: depot-anchored demand-weighted path substitution
        if depot_candidates:
            candidates = []
            for i, node in enumerate(route):
                node_int = int(node)
                if node_int in depot_candidates:
                    candidates.append(i)
            if candidates:
                cut_u = random.choice(candidates)
                cut_v = random.randint(max(0, cut_u + 2), min(len(route) - 1, cut_u + 6))
            else:
                cut_u = random.randint(0, len(route) - 3)
                cut_v = random.randint(cut_u + 2, len(route) - 1)
        else:
            cut_u = random.randint(0, len(route) - 3)
            cut_v = random.randint(cut_u + 2, len(route) - 1)
        u, v = int(route[cut_u]), int(route[cut_v])

        try:
            def demand_weight(a, b, d):
                dem_b = env.tourist_map.get(b, 0)
                return d.get('length', 1.0) / (1.0 + ε * dem_b)
            subpath = nx.shortest_path(env.G, u, v, weight=demand_weight, method='dijkstra')
            new_route = route[:cut_u + 1] + subpath[1:-1] + route[cut_v:]
            # Validate connectivity
            valid = True
            for i in range(len(new_route) - 1):
                a, b = int(new_route[i]), int(new_route[i + 1])
                if not env.G.has_edge(a, b):
                    valid = False
                    break
            if valid:
                ind[idx] = new_route
        except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
            pass

        return (ind, )

    except Exception as e:
        return (copy.deepcopy(input_ind), )