#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Structure for a node in the graph
struct Node {
    int value;
    vector<Node*> neighbors;
    bool visited;
    Node(int val) : value(val), visited(false) {}
};

// Breadth First Search on a Tree
void parallelBFS_Tree(Node* root) {
    queue<Node*> q;
    root->visited = true;
    q.push(root);

    while (!q.empty()) {
        #pragma omp parallel for
        for (int i = 0; i < q.size(); ++i) {
            Node* current = q.front();
            q.pop();
            cout << current->value << " ";

            // Enqueue all unvisited neighbors
            for (Node* neighbor : current->neighbors) {
                if (!neighbor->visited) {
                    neighbor->visited = true;
                    q.push(neighbor);
                }
            }
        }
    }
}

// Depth First Search on a Tree
void parallelDFS_Tree(Node* node) {
    if (node == nullptr)
        return;
    
    node->visited = true;
    cout << node->value << " ";

    #pragma omp parallel for
    for (int i = 0; i < node->neighbors.size(); ++i) {
        if (!node->neighbors[i]->visited)
            parallelDFS_Tree(node->neighbors[i]);
    }
}

int main() {
    // Creating a tree
    Node* root = new Node(1);
    Node* node2 = new Node(2);
    Node* node3 = new Node(3);
    Node* node4 = new Node(4);
    Node* node5 = new Node(5);

    root->neighbors = {node2, node3};
    node2->neighbors = {root, node4, node5};
    node3->neighbors = {root};
    node4->neighbors = {node2};
    node5->neighbors = {node2};

    // Parallel BFS on tree
    cout << "Parallel BFS on Tree: ";
    parallelBFS_Tree(root);
    cout << endl;

    // Resetting visited flags
    root->visited = false;
    node2->visited = false;
    node3->visited = false;
    node4->visited = false;
    node5->visited = false;

    // Parallel DFS on tree
    cout << "Parallel DFS on Tree: ";
    parallelDFS_Tree(root);
    cout << endl;

    return 0;
}
