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

        # 3. Compute global uncovered demand: high-demand nodes not covered by *any* route in population
        all_covered = set()
        for r in ind:
            all_covered.update(r)
        all_nodes = list(env.tourist_map.keys())
        high_demand_nodes = {n for n in all_nodes if env.tourist_map.get(n, 0) >= 1}
        uncovered_demand = high_demand_nodes - all_covered

        # 4. Unified visual-demand edge cost: length / (PI - NI + ε), ε = 1e-6
        ε = 1e-6

        def visual_demand_weight(u_, v_, d):
            length = d.get('length', 1.0)
            pi = d.get('PI', 0.0)
            ni = d.get('NI', 0.0)
            denominator = pi - ni + ε
            return length / denominator if denominator > 0 else float('inf')

        # 5. Proactive depot constraint: ensure start/end are valid depots (tourist=0)
        depot_candidates = [n for n in all_nodes if env.G.nodes[n].get('tourist', 0) == 0]
        if not depot_candidates:
            return (ind,)

        # 6. Score edges by novelty + visual contribution: use cached Dijkstra with cutoff
        # Precompute route set once
        route_set = set(route)

        # Build edge scores: novelty (uncovered demand reachable) + visual contribution
        edge_scores = []
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            score = 0.0
            # Visual contribution: normalized by unified cost
            if env.G.has_edge(u, v):
                w = visual_demand_weight(u, v, env.G.edges[u, v])
                if w != float('inf'):
                    score += 1.0 / (w + 1e-6)
            # Novelty: count high-demand nodes newly reachable *via this edge* (within bounded radius)
            # Use cached shortest paths with max hops = 5 to prevent blowup
            try:
                # Limit path search depth via hop-bound Dijkstra (not distance)
                dists = nx.single_source_dijkstra_path_length(
                    env.G, u, cutoff=5, weight='length'
                )
                for n in high_demand_nodes:
                    if n in dists and n not in route_set:
                        score += 0.5 * env.tourist_map.get(n, 0)
            except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
                pass
            edge_scores.append(score)

        # 7. Bounded, demand-guided pathfinding: select worst-scoring edge window for ruin
        window_size = 3
        if len(edge_scores) < window_size:
            return (ind,)
        min_score_sum = float('inf')
        best_start = 0
        for start in range(len(edge_scores) - window_size + 1):
            s = sum(edge_scores[start:start + window_size])
            if s < min_score_sum:
                min_score_sum = s
                best_start = start

        ruin_start_idx = best_start
        ruin_end_idx = best_start + window_size + 1
        if ruin_end_idx > len(route):
            ruin_end_idx = len(route)
        if ruin_end_idx - ruin_start_idx < 2:
            return (ind,)

        u_ruin = route[ruin_start_idx]
        v_ruin = route[ruin_end_idx - 1]

        # Ensure u_ruin, v_ruin are valid and connected
        if not (env.G.has_node(u_ruin) and env.G.has_node(v_ruin)):
            return (ind,)

        # 8. Recreate using unified visual-demand cost + bounded Dijkstra
        recreated_path = None
        max_hops = 10
        try:
            # Use custom weight function with hop limit via manual Dijkstra
            def bounded_dijkstra(source, target, max_hops):
                import heapq
                if source == target:
                    return [source]
                pq = [(0.0, 0, source, [source])]  # (cost, hops, node, path)
                visited = {}
                while pq:
                    cost, hops, node, path = heapq.heappop(pq)
                    if node == target:
                        return path
                    if hops >= max_hops:
                        continue
                    if node in visited and visited[node] <= hops:
                        continue
                    visited[node] = hops
                    for neighbor in env.G.neighbors(node):
                        if env.G.has_edge(node, neighbor):
                            w = visual_demand_weight(node, neighbor, env.G.edges[node, neighbor])
                            if w == float('inf'):
                                continue
                            new_cost = cost + w
                            new_hops = hops + 1
                            new_path = path + [neighbor]
                            heapq.heappush(pq, (new_cost, new_hops, neighbor, new_path))
                return None

            recreated_path = bounded_dijkstra(u_ruin, v_ruin, max_hops)
        except Exception:
            pass

        # Fallback: standard Dijkstra with visual-demand weight, with timeout safety
        if recreated_path is None:
            try:
                recreated_path = nx.shortest_path(
                    env.G, u_ruin, v_ruin, weight=visual_demand_weight
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, nx.NetworkXError):
                # Ultimate fallback: length-only, with hop limit
                try:
                    recreated_path = nx.shortest_path(
                        env.G, u_ruin, v_ruin, weight='length'
                    )
                    # Trim if too long
                    if len(recreated_path) > 20:
                        recreated_path = recreated_path[:20]
                except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError, nx.NetworkXError):
                    return (ind,)

        # 9. Enforce depot constraints proactively: ensure start/end of recreated_path are depots
        if recreated_path:
            # Adjust start if not depot
            if env.G.nodes.get(recreated_path[0], {}).get('tourist', 0) == 1:
                # Find nearest depot within 3 hops
                found_depot = False
                for hop in range(1, 4):
                    try:
                        neighbors = list(nx.single_source_shortest_path_length(env.G, recreated_path[0], cutoff=hop).keys())
                        depot_neighbors = [n for n in neighbors if env.G.nodes.get(n, {}).get('tourist', 0) == 0]
                        if depot_neighbors:
                            nearest_depot = min(depot_neighbors, key=lambda x: env.distance_map.get((recreated_path[0], x), float('inf')))
                            if env.distance_map.get((recreated_path[0], nearest_depot), float('inf')) != float('inf'):
                                recreated_path = [nearest_depot] + recreated_path
                                found_depot = True
                                break
                    except:
                        pass
                if not found_depot:
                    recreated_path = [depot_candidates[0]] + recreated_path
            # Adjust end if not depot
            if env.G.nodes.get(recreated_path[-1], {}).get('tourist', 0) == 1:
                found_depot = False
                for hop in range(1, 4):
                    try:
                        neighbors = list(nx.single_source_shortest_path_length(env.G, recreated_path[-1], cutoff=hop).keys())
                        depot_neighbors = [n for n in neighbors if env.G.nodes.get(n, {}).get('tourist', 0) == 0]
                        if depot_neighbors:
                            nearest_depot = min(depot_neighbors, key=lambda x: env.distance_map.get((recreated_path[-1], x), float('inf')))
                            if env.distance_map.get((recreated_path[-1], nearest_depot), float('inf')) != float('inf'):
                                recreated_path = recreated_path + [nearest_depot]
                                found_depot = True
                                break
                    except:
                        pass
                if not found_depot:
                    recreated_path = recreated_path + [depot_candidates[0]]

        # 10. Assemble new route
        if recreated_path and len(recreated_path) >= 2:
            new_route = route[:ruin_start_idx+1] + recreated_path[1:-1] + route[ruin_end_idx:]
            # Final validation: ensure route starts/ends at depots
            if new_route and env.G.nodes.get(new_route[0], {}).get('tourist', 0) == 1:
                new_route = [depot_candidates[0]] + new_route
            if new_route and env.G.nodes.get(new_route[-1], {}).get('tourist', 0) == 1:
                new_route = new_route + [depot_candidates[0]]
            ind[idx] = new_route

        return (ind,)

    except Exception as e:
        return (copy.deepcopy(input_ind),)