def mt_operator(input_ind, env):
    try:
        # 1. Deep copy to avoid modifying original
        ind = copy.deepcopy(input_ind)

        # 2. Select a route randomly
        if not ind:
            return (ind,)
        idx = random.randint(0, len(ind) - 1)
        route = ind[idx]
        if len(route) < 4:
            return (ind,)

        # Precompute global uncovered high-demand nodes (tourist=1)
        all_high_demand = [n for n in env.tourist_map if env.tourist_map[n] == 1]
        if not all_high_demand:
            # Fallback: visual-weighted segment replacement with bounded pathfinding
            cut_u = random.randint(0, len(route) - 3)
            cut_v = random.randint(cut_u + 2, len(route) - 1)
            u, v = route[cut_u], route[cut_v]

            def weight_func(u_, v_, d):
                length = d.get('length', 1.0)
                pi = d.get('PI', 0.0)
                ni = d.get('NI', 0.0)
                score = max(pi - ni, 0.0) + 1e-6
                return length / score

            try:
                subpath = nx.shortest_path(env.G, int(u), int(v), weight=weight_func, method='dijkstra')
                if len(subpath) > 50:
                    raise nx.NetworkXNoPath
                new_route = route[:cut_u+1] + subpath[1:-1] + route[cut_v:]
                ind[idx] = new_route
            except (nx.NetworkXNoPath, nx.NodeNotFound, Exception):
                pass
            return (ind,)

        # Compute globally uncovered demand
        covered_by_any = set()
        for r in ind:
            covered_by_any.update(r)
        deficit_nodes = [n for n in all_high_demand if n not in covered_by_any]
        
        # Prioritize marginal gain: score each uncovered node by demand-weighted distance to nearest depot + novelty
        depot_nodes = [n for n in env.G.nodes() if env.G.nodes[n].get('tourist', 0) == 0 and 
                      (n == env.init_route['start_end'][0] or n == env.init_route['start_end'][1])]
        if not depot_nodes:
            depot_nodes = [route[0], route[-1]]

        # Precompute demand-weighted distances from all nodes to all depots (cache-efficient)
        demand_dist_scores = {}
        for node in deficit_nodes:
            min_weighted_dist = float('inf')
            for depot in depot_nodes:
                try:
                    # Use unified cost for pathfinding: length / (1 + ε·demand + δ·max(PI−NI,0))
                    def unified_cost(u_, v_, d):
                        length = d.get('length', 1.0)
                        dest_demand = env.tourist_map.get(int(v_), 0)
                        pi = d.get('PI', 0.0)
                        ni = d.get('NI', 0.0)
                        return length / (1.0 + 1e-3 * dest_demand + 1e-6 * max(pi - ni, 0.0) + 1e-9)

                    if env.G.has_edge(int(depot), int(node)):
                        cost = env.G.edges[int(depot), int(node)]['length']
                        weighted_dist = cost / (1.0 + 1e-3 * env.tourist_map.get(int(node), 0) + 
                                               1e-6 * max(env.G.edges[int(depot), int(node)].get('PI', 0.0) - 
                                                         env.G.edges[int(depot), int(node)].get('NI', 0.0), 0.0) + 1e-9)
                        min_weighted_dist = min(min_weighted_dist, weighted_dist)
                    else:
                        path = nx.shortest_path(env.G, int(depot), int(node), weight=unified_cost, method='dijkstra')
                        if len(path) > 100:
                            continue
                        dist = sum(env.G.edges[path[i], path[i+1]]['length'] 
                                  for i in range(len(path)-1))
                        weighted_dist = dist / (1.0 + 1e-3 * env.tourist_map.get(int(node), 0) + 
                                               1e-6 * max(sum(env.G.edges[path[j], path[j+1]].get('PI', 0.0) 
                                                             for j in range(len(path)-1)) - 
                                                         sum(env.G.edges[path[j], path[j+1]].get('NI', 0.0) 
                                                             for j in range(len(path)-1)), 0.0) + 1e-9)
                        min_weighted_dist = min(min_weighted_dist, weighted_dist)
                except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, Exception):
                    continue
            # Novelty bonus: inverse of how many times this node appears in population routes
            novelty_bonus = 1.0
            for r in ind:
                if node in r:
                    novelty_bonus *= 0.5
            demand_dist_scores[node] = -min_weighted_dist * novelty_bonus  # higher = better

        # Sort by marginal gain (descending)
        sorted_deficit = sorted(deficit_nodes, key=lambda x: demand_dist_scores.get(x, -float('inf')), reverse=True)
        
        # Attempt insertion of highest-scoring uncovered node
        inserted = False
        if sorted_deficit:
            ghost_node = sorted_deficit[0]

            # Score all valid internal insertion points using unified edge cost
            best_score = -float('inf')
            best_pair = None

            for i in range(1, len(route) - 1):
                u, v = route[i-1], route[i]
                try:
                    def unified_cost(u_, v_, d):
                        length = d.get('length', 1.0)
                        dest_demand = env.tourist_map.get(int(v_), 0)
                        pi = d.get('PI', 0.0)
                        ni = d.get('NI', 0.0)
                        return length / (1.0 + 1e-3 * dest_demand + 1e-6 * max(pi - ni, 0.0) + 1e-9)

                    # Try direct edge existence first
                    if env.G.has_edge(int(u), int(ghost_node)) and env.G.has_edge(int(ghost_node), int(v)):
                        cost_u_g = env.G.edges[int(u), int(ghost_node)]['length']
                        cost_g_v = env.G.edges[int(ghost_node), int(v)]['length']
                        score = -(cost_u_g + cost_g_v)  # maximize negative cost
                    else:
                        # Bounded shortest paths
                        path_u_g = nx.shortest_path(env.G, int(u), int(ghost_node), weight=unified_cost, method='dijkstra')
                        path_g_v = nx.shortest_path(env.G, int(ghost_node), int(v), weight=unified_cost, method='dijkstra')
                        if len(path_u_g) > 50 or len(path_g_v) > 50:
                            continue
                        cost_u_g = sum(env.G.edges[path_u_g[j], path_u_g[j+1]]['length'] 
                                      for j in range(len(path_u_g)-1))
                        cost_g_v = sum(env.G.edges[path_g_v[j], path_g_v[j+1]]['length'] 
                                      for j in range(len(path_g_v)-1))
                        score = -(cost_u_g + cost_g_v)

                    if score > best_score:
                        best_score = score
                        best_pair = (i-1, i)
                except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, Exception):
                    continue

            if best_pair is not None:
                i_before, i_after = best_pair
                u, v = route[i_before], route[i_after]

                try:
                    def unified_cost(u_, v_, d):
                        length = d.get('length', 1.0)
                        dest_demand = env.tourist_map.get(int(v_), 0)
                        pi = d.get('PI', 0.0)
                        ni = d.get('NI', 0.0)
                        return length / (1.0 + 1e-3 * dest_demand + 1e-6 * max(pi - ni, 0.0) + 1e-9)

                    if env.G.has_edge(int(u), int(ghost_node)) and env.G.has_edge(int(ghost_node), int(v)):
                        detour = [u, ghost_node, v]
                    else:
                        path_u_g = nx.shortest_path(env.G, int(u), int(ghost_node), weight=unified_cost, method='dijkstra')
                        path_g_v = nx.shortest_path(env.G, int(ghost_node), int(v), weight=unified_cost, method='dijkstra')
                        if len(path_u_g) > 50 or len(path_g_v) > 50:
                            raise nx.NetworkXNoPath
                        detour = path_u_g[:-1] + path_g_v

                    new_route = route[:i_before+1] + detour[1:-1] + route[i_after:]
                    ind[idx] = new_route
                    inserted = True
                except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, Exception):
                    pass

        if inserted:
            return (ind,)

        # Ruin-and-Recreate with novelty-aware reconstruction
        route_set = set(route)
        covered_in_route = {n for n in route if env.tourist_map.get(n, 0) == 1}
        unvisited_demand = [n for n in all_high_demand if n not in route_set]

        # Score edges by novelty + visual contribution + marginal demand gain
        edge_scores = []
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            if not env.G.has_edge(int(u), int(v)):
                edge_scores.append(-1e6)
                continue

            d = env.G.edges[int(u), int(v)]
            length = d.get('length', 1.0)
            pi = d.get('PI', 0.0)
            ni = d.get('NI', 0.0)
            visual_contrib = (pi - ni) / max(length, 1e-6)
            satisfy_contrib = 1.0 if (env.tourist_map.get(v, 0) == 1 and v not in covered_in_route) else 0.0
            # Novelty: penalize edges frequently used across population
            freq_penalty = 0.0
            for r in ind:
                for j in range(len(r)-1):
                    if r[j] == u and r[j+1] == v:
                        freq_penalty += 0.1
            score = 0.4 * visual_contrib + 0.4 * satisfy_contrib - 0.2 * freq_penalty
            edge_scores.append(score)

        # Find worst contiguous window (3–7 edges) by avg score
        best_avg = float('inf')
        best_window = None
        n_edges = len(edge_scores)
        if n_edges >= 3:
            for L in range(3, min(8, n_edges + 1)):
                for start in range(n_edges - L + 1):
                    avg = sum(edge_scores[start:start+L]) / L
                    if avg < best_avg:
                        best_avg = avg
                        best_window = (start, start + L)

        if best_window is not None:
            start_edge_idx, end_edge_idx = best_window
            u = route[start_edge_idx]
            v = route[end_edge_idx + 1]

            prefix = route[:start_edge_idx + 1]
            suffix = route[end_edge_idx + 1:]

            # Reconstruct using top unvisited demand nodes as intermediaries — with bounded search
            candidate_path = None
            try:
                # Try top-3 unvisited demand nodes, with step limit
                for mid in unvisited_demand[:3]:
                    try:
                        def unified_cost(u_, v_, d):
                            length = d.get('length', 1.0)
                            dest_demand = env.tourist_map.get(int(v_), 0)
                            pi = d.get('PI', 0.0)
                            ni = d.get('NI', 0.0)
                            return length / (1.0 + 1e-3 * dest_demand + 1e-6 * max(pi - ni, 0.0) + 1e-9)

                        p1 = nx.shortest_path(env.G, int(u), int(mid), weight=unified_cost, method='dijkstra')
                        p2 = nx.shortest_path(env.G, int(mid), int(v), weight=unified_cost, method='dijkstra')
                        if len(p1) > 50 or len(p2) > 50:
                            continue
                        full_path = p1[:-1] + p2
                        if candidate_path is None or len(full_path) < len(candidate_path):
                            candidate_path = full_path
                    except (nx.NetworkXNoPath, nx.NodeNotFound, Exception):
                        continue

                if candidate_path is None:
                    # Fallback: direct path with unified cost
                    candidate_path = nx.shortest_path(env.G, int(u), int(v), 
                                                    weight=lambda u_, v_, d: d.get('length', 1.0) / 
                                                    (1.0 + 1e-3 * env.tourist_map.get(int(v_), 0) + 
                                                     1e-6 * max(d.get('PI', 0.0) - d.get('NI', 0.0), 0.0) + 1e-9),
                                                    method='dijkstra')
                    if len(candidate_path) > 100:
                        raise nx.NetworkXNoPath
            except (nx.NetworkXNoPath, nx.NodeNotFound, Exception):
                try:
                    candidate_path = nx.shortest_path(env.G, int(u), int(v), method='dijkstra')
                    if len(candidate_path) > 100:
                        raise nx.NetworkXNoPath
                except (nx.NetworkXNoPath, nx.NodeNotFound, Exception):
                    return (ind,)

            inner = candidate_path[1:-1]
            new_route = prefix + inner + suffix
            ind[idx] = new_route
            return (ind,)

        # Stochastic operator selection for exploration balance
        op_choice = random.random()
        if op_choice < 0.3:
            # Random node swap within route (preserving depots)
            if len(route) > 5:
                i, j = random.sample(range(1, len(route)-1), 2)
                new_route = route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                ind[idx] = new_route
        elif op_choice < 0.6:
            # Random edge flip (reverse segment)
            if len(route) > 5:
                i = random.randint(1, len(route)-3)
                j = random.randint(i+2, len(route)-2)
                new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
                ind[idx] = new_route
        else:
            # Visual-guided local perturbation: replace middle node with best neighbor
            if len(route) > 5:
                mid_idx = random.randint(2, len(route)-3)
                u, v = route[mid_idx-1], route[mid_idx+1]
                candidates = list(env.G.neighbors(int(route[mid_idx])))
                if candidates:
                    best_candidate = None
                    best_score = -float('inf')
                    for cand in candidates:
                        if cand == u or cand == v:
                            continue
                        try:
                            d_uv = env.G.edges[int(u), int(cand)]['length'] + env.G.edges[int(cand), int(v)]['length']
                            pi_cand = env.G.adj[int(u)][int(cand)].get('PI', 0.0) + env.G.adj[int(cand)][int(v)].get('PI', 0.0)
                            ni_cand = env.G.adj[int(u)][int(cand)].get('NI', 0.0) + env.G.adj[int(cand)][int(v)].get('NI', 0.0)
                            score = (pi_cand - ni_cand) / (d_uv + 1e-6)
                            if score > best_score:
                                best_score = score
                                best_candidate = cand
                        except (KeyError, Exception):
                            continue
                    if best_candidate is not None:
                        new_route = route[:]
                        new_route[mid_idx] = best_candidate
                        ind[idx] = new_route

        return (ind,)

    except Exception as e:
        return (copy.deepcopy(input_ind),)