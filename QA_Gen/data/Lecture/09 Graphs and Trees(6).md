# 09 Graphs and Trees (图与树)

## 1. Graphs and Graph Models (图与图模型)
- **Graph (图)**: A graph $G=(V,E)$ consists of $V$, a nonempty set of **vertices (顶点)** (or nodes), and $E$, a set of **edges (边)**. Each edge has one or two vertices associated with it, called its **endpoints (端点)**.
- **Simple Graph (简单图)**: A graph in which each edge connects two different vertices and no two edges connect the same pair of vertices.
- **Multigraph (重图)**: A graph that may have **multiple edges (重边)** connecting the same pair of vertices.
- **Pseudograph (伪图)**: A graph that may include **loops (自环)**, and possibly multiple edges.
- **Directed Graph (有向图/digraph)**: $G=(V,E)$ consists of a nonempty set of vertices $V$, and a set of directed edges (arcs) $E$. Each directed edge is associated with an ordered pair of vertices $(u, v)$, starting at $u$ (initial vertex) and ending at $v$ (terminal vertex).

## 2. Graph Terminology and Operations (图的术语与操作)
- **Adjacent (相邻)**: Two vertices $u$ and $v$ in an undirected graph are adjacent if they are endpoints of an edge $e$. The edge $e$ is **incident (关联)** with $u$ and $v$.
- **Neighborhood (邻域)**: $N(v)$ is the set of all neighbors of vertex $v$. For a subset $A \subseteq V$, $N(A) = \cup_{v \in A} N(v)$.
- **Degree (度)**: $deg(v)$ is the number of edges incident with a vertex (a loop contributes twice). A vertex of degree 0 is **isolated**; a vertex of degree 1 is **pendant**.
- **The Handshaking Theorem (握手定理)**: If $G=(V,E)$ is an undirected graph with $e$ edges, then $2e = \sum_{v \in V} deg(v)$.
  - *Corollary*: An undirected graph has an even number of vertices of odd degree.
- **Directed Degree**: **In-degree (入度)** $deg^-(v)$ and **out-degree (出度)** $deg^+(v)$.
  - *Theorem*: If $G=(V,E)$ is a directed graph with $e$ edges, then $e = \sum_{v \in V} deg^-(v) = \sum_{v \in V} deg^+(v)$.
- **Subgraph (子图)**: $H=(W,F)$ is a subgraph of $G=(V,E)$ if $W \subseteq V$ and $F \subseteq E$. It is an **induced subgraph (导出子图)** by $W$ if $F$ contains an edge in $E$ if and only if both endpoints are in $W$.
- **Graph Unions**: The union of two simple graphs $G_1=(V_1,E_1)$ and $G_2=(V_2,E_2)$ is $G_1 \cup G_2 = (V_1 \cup V_2, E_1 \cup E_2)$.

## 3. Special Types of Graphs (特殊类型的图)
- **Complete Graph (完全图)** $K_n$: A simple graph that contains exactly one edge between each pair of distinct vertices.
- **Cycle (环)** $C_n$ ($n \ge 3$): Consists of $n$ vertices and edges forming a single closed loop.
- **Wheel (轮)** $W_n$: Obtained by adding an additional vertex to a cycle $C_n$ and connecting this new vertex to each of the $n$ vertices in $C_n$.
- **n-Cube (超立方体)** $Q_n$: A graph that has vertices representing the $2^n$ bit strings of length $n$. Two vertices are adjacent iff their bit strings differ in exactly one position.
- **Bipartite Graph (二分图)**: A simple graph $G=(V,E)$ such that $V$ can be partitioned into two disjoint sets $V_1, V_2$, and every edge connects one vertex in $V_1$ to one in $V_2$.
  - *Theorem*: A simple graph is bipartite if and only if it is possible to assign 2 colors to its vertices such that no two adjacent vertices have the same color.
- **Complete Bipartite Graph (完全二分图)** $K_{m,n}$: The vertex set is partitioned into two subsets of $m$ and $n$ vertices, with an edge between two vertices iff one is in the first subset and the other in the second.

## 4. Matchings (匹配)
- **Matching (匹配)** $M$: A subset of the edge set $E$ such that no two edges are incident with the same vertex.
- **Maximum matching (最大匹配)**: A matching with the largest number of edges.
- **Complete matching (完全匹配)** from $V_1$ to $V_2$: Every vertex in $V_1$ is the endpoint of an edge in the matching ($|M| = |V_1|$).
- **Hall's Marriage Theorem (霍尔婚姻定理)**: The bipartite graph $G=(V,E)$ with bipartition $(V_1, V_2)$ has a complete matching from $V_1$ to $V_2$ if and only if $|N(A)| \ge |A|$ for all subsets $A \subseteq V_1$.

## 5. Graph Representation and Isomorphism (图的表示与同构)
- **Adjacency Lists (邻接表)**: Specifies the vertices that are adjacent to each vertex.
- **Adjacency Matrices (邻接矩阵)** $A$: An $n \times n$ matrix where the $(i, j)$-th entry is the number of edges connecting $v_i$ and $v_j$ (for simple graphs, it's a 0-1 matrix).
- **Incidence Matrices (关联矩阵)** $M$: An $n \times m$ matrix where entry is 1 if edge $e_j$ is incident with $v_i$, otherwise 0.
- **Isomorphism (同构)**: $G_1=(V_1,E_1)$ and $G_2=(V_2,E_2)$ are isomorphic if there exists a bijective function $f$ from $V_1$ to $V_2$ such that $a$ and $b$ are adjacent in $G_1$ iff $f(a)$ and $f(b)$ are adjacent in $G_2$.
  - Checking graph invariants (e.g., number of vertices, edges, degree sequence) can help determine if graphs are non-isomorphic.

## 6. Connectivity (连通性)
- **Path (路径)**: A sequence of edges passing through vertices. A path is a **circuit (回路)** if it begins and ends at the same vertex. A path is **simple (简单)** if it does not contain the same edge more than once.
- **Connected (连通)**: An undirected graph is connected if there is a path between every pair of distinct vertices.
- **Strongly Connected (强连通)**: A directed graph has a path from $a$ to $b$ and $b$ to $a$ for all pairs $a, b$. **Weakly connected (弱连通)** if the underlying undirected graph is connected.
- **Connected Component (连通分量)**: A maximal connected subgraph.
- **Cut Vertices (割点) & Cut Edges/Bridges (割边/桥)**: Vertices or edges whose removal produces a subgraph with more connected components.
- **Counting Paths**: The number of different paths of length $r$ from $v_i$ to $v_j$ equals the $(i, j)$-th entry of the matrix $A^r$ (where $A$ is the adjacency matrix).

## 7. Shortest-Path Problems (最短路径问题)
- **Weighted Graphs (加权图)**: Graphs that have a weight assigned to each edge.
- **Dijkstra's Algorithm (迪杰斯特拉算法)**: Finds the shortest path between two vertices in a connected simple weighted graph (with non-negative weights).
  - *Time Complexity*: $O(n^2)$, which can be improved to $O((m+n)\log n)$ via a priority queue for $m$ edges.

## 8. Euler and Hamilton Paths (欧拉与哈密顿路径)
- **Euler Circuit (欧拉回路)**: A simple circuit containing every edge of $G$. 
  - *Theorem*: Exists if and only if the graph is connected and **each of its vertices has an even degree**.
- **Euler Path (欧拉路径)**: A simple path containing every edge.
  - *Theorem*: Exists but no Euler circuit if and only if the graph has **exactly 2 vertices of odd degree**.
- **Hamilton Path/Circuit (哈密顿路径/回路)**: A simple path/circuit that passes through every vertex exactly once. (This is an NP-complete problem).
  - *Dirac's Theorem*: If $n \ge 3$ and $deg(v) \ge n/2$ for all vertices, then $G$ has a Hamilton circuit.
  - *Ore's Theorem*: If $n \ge 3$ and $deg(u)+deg(v) \ge n$ for every pair of nonadjacent vertices, then $G$ has a Hamilton circuit.

## 9. Planar Graphs (平面图)
- **Planar Graph**: A graph that can be drawn in the plane without any edges crossing.
- **Euler's Formula (欧拉公式)**: For a connected planar simple graph with $e$ edges, $v$ vertices, and $r$ regions, **$r = e - v + 2$**.
- **Corollaries**:
  1. If $v \ge 3$, then $e \le 3v - 6$.
  2. A connected planar simple graph has a vertex of degree $\le 5$.
  3. If $v \ge 3$ and there are no circuits of length 3 (no triangles), then $e \le 2v - 4$.
- **Kuratowski's Theorem**: A graph is nonplanar if and only if it contains a subgraph homeomorphic to $K_{3,3}$ or $K_5$.

## 10. Graph Coloring (图染色)
- **Chromatic Number (染色数)** $\chi(G)$: The least number of colors needed to color vertices so that no two adjacent vertices share the same color.
- **The Four-Color Theorem**: For any simple planar graph $G$, $\chi(G) \le 4$. (Proved in 1976 via computer-aided analysis).
- *Note*: $\chi(K_n) = n$, $\chi(K_{m,n}) = 2$.

## 11. Trees (树)
- **Tree**: A connected undirected graph without simple circuits.
  - *Theorem*: An undirected graph is a tree iff there is a **unique simple path** between any pair of vertices.
  - *Theorem*: A tree with $n$ vertices has **$n - 1$ edges**.
- **Rooted Tree (有根树)**: One vertex is designated as the root. Terminology includes parent, child, siblings, ancestors, descendants, leaves (degree 1), and internal vertices.
- **m-ary Tree (m元树)**: Every internal vertex has $\le m$ children. It is a **full m-ary tree (满m元树)** if every internal vertex has exactly $m$ children (e.g., Binary Tree when $m=2$).
  - For a full m-ary tree: $n = mi + 1$ (vertices) and $l = (m-1)i + 1$ (leaves).
  - **Height $h$**: The maximum level. A tree is **balanced (平衡)** if all leaves are at levels $h$ or $h-1$.
  - Number of leaves bounded by $l \le m^h$. Height bounded by $h \ge \lceil \log_m l \rceil$.
- **Tree Traversal (遍历)**:
  - **Preorder (前序)**: Root $\rightarrow$ Left $\rightarrow$ Right (used for Prefix / Polish notation).
  - **Inorder (中序)**: Left $\rightarrow$ Root $\rightarrow$ Right (used for Infix notation, requires parentheses).
  - **Postorder (后序)**: Left $\rightarrow$ Right $\rightarrow$ Root (used for Postfix / Reverse Polish notation).
- **Spanning Tree (生成树)**: A subgraph that is a tree containing every vertex of $G$. Can be found via **DFS (深度优先搜索)** or **BFS (广度优先搜索)**.
- **Minimum Spanning Tree (最小生成树)**: A spanning tree with the smallest possible sum of edge weights.
  - **Prim's Algorithm**: Builds the tree node by node. Time complexity: $O(|E|\log|V|)$.
  - **Kruskal's Algorithm**: Builds the tree edge by edge (sorting edges first). Time complexity: $O(|E|\log|E|)$.