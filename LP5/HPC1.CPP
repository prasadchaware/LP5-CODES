#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

void parallel_bfs(vector<vector<int>>& graph, int start) {
    int num_nodes = graph.size();
    vector<bool> visited(num_nodes, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int current = q.front();
        q.pop();
        cout << "Visited: " << current << endl;
        #pragma omp parallel for shared(graph, visited, q)
        for (int neighbor : graph[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

void dfs_recursive(vector<vector<int>>& graph, int node, vector<bool>& visited) {
    visited[node] = true;
    cout << "Visited: " << node << endl;
    #pragma omp parallel for shared(graph, visited)
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            dfs_recursive(graph, neighbor, visited);
        }
    }
}

void parallel_dfs(vector<vector<int>>& graph, int start) {
    int num_nodes = graph.size();
    vector<bool> visited(num_nodes, false);
    dfs_recursive(graph, start, visited);
}

int main() {
    vector<vector<int>> graph = {
        {1, 2},
        {0, 3, 4},
        {0, 5},
        {1},
        {1},
        {2}
    };

    int start_node = 0;

    cout << "BFS Traversal:" << endl;
    parallel_bfs(graph, start_node);

    cout << "DFS Traversal:" << endl;
    parallel_dfs(graph, start_node);

    return 0;
}
