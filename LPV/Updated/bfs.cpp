#include <iostream>
#include <vector>
#include <queue>
#include <ctime>
#include <omp.h>
using namespace std;

class Node {
public:
    Node* left;
    Node* right;
    int data;
};

class BreadthFS {
public:
    Node* insert(Node* root, int data);
    void bfs(Node* head);
};

Node* BreadthFS::insert(Node* root, int data) {
    if (!root) {
        root = new Node;
        root->left = NULL;
        root->right = NULL;
        root->data = data;
        return root;
    }

    queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        Node* temp = q.front();
        q.pop();

        if (temp->left == NULL) {
            temp->left = new Node;
            temp->left->left = NULL;
            temp->left->right = NULL;
            temp->left->data = data;
            return root;
        } else {
            q.push(temp->left);
        }

        if (temp->right == NULL) {
            temp->right = new Node;
            temp->right->left = NULL;
            temp->right->right = NULL;
            temp->right->data = data;
            return root;
        } else {
            q.push(temp->right);
        }
    }

    return root;
}

void BreadthFS::bfs(Node* head) {
    queue<Node*> q;
    q.push(head);

    int qSize;

    while (!q.empty()) {
        qSize = q.size();
        #pragma omp parallel for
        for (int i = 0; i < qSize; i++) {
            Node* currNode;
            #pragma omp critical
            {
                currNode = q.front();
                q.pop();
                cout << "\t" << currNode->data;
            }
            #pragma omp critical
            {
                if (currNode->left)
                    q.push(currNode->left);
                if (currNode->right)
                    q.push(currNode->right);
            }
        }
    }
}

// Function to perform BFS from a given vertex
void bfs(int startVertex, vector<bool>& visited, vector<vector<int>>& graph) {
    // Create a queue for BFS
    queue<int> q;
    // Mark the start vertex as visited and enqueue it
    visited[startVertex] = true;
    q.push(startVertex);
    // Loop until the queue is empty
    while (!q.empty()) {
        // Dequeue a vertex from the queue
        int v = q.front();
        q.pop();
        // Enqueue all adjacent vertices that are not visited
        #pragma omp parallel for
        for (int i = 0; i < graph[v].size(); i++) {
            int u = graph[v][i];
            #pragma omp critical
            {
                if (!visited[u]) {
                    visited[u] = true;
                    q.push(u);
                }
            }
        }
    }
}

// Parallel Breadth-First Search
void parallelBFS(vector<vector<int>>& graph, int numCores) {
    int numVertices = graph.size();
    vector<bool> visited(numVertices, false); // Keep track of visited vertices
    double startTime = omp_get_wtime(); // Start timer
    // Perform BFS from all unvisited vertices using the specified number of cores
    #pragma omp parallel for num_threads(numCores)
    for (int v = 0; v < numVertices; v++) {
        if (!visited[v]) {
            bfs(v, visited, graph);
        }
    }
    double endTime = omp_get_wtime(); // End timer
    cout << "Number of cores used: " << numCores << endl;
    cout << "Time taken: " << endTime - startTime << " seconds" << endl;
    cout << "------------------------" << endl;
}

int main() {
    Node* root = NULL;
    int data;
    char ans;

    do {
        cout << "\nEnter data => ";
        cin >> data;

        BreadthFS bfsObj;
        root = bfsObj.insert(root, data);

        cout << "Do you want to insert one more node? (y/n): ";
        cin >> ans;
    } while (ans == 'y' || ans == 'Y');

    BreadthFS bfsObj;
    cout << "BFS traversal: ";
    bfsObj.bfs(root);
    cout << endl;

    // Generate a random graph with 10,000 vertices and 50,000 edges
    int numVertices = 10000;
    int numEdges = 50000;
    vector<vector<int>> graph(numVertices);
    srand(time(0));
    for (int i = 0; i < numEdges; i++) {
        int u = rand() % numVertices;
        int v = rand() % numVertices;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    // Array containing the number of cores
    int numCoresArr[] = { 1, 2, 3, 4, 5, 6, 7, 8 };

    // Loop over different numbers of cores and execute parallel BFS
    for (int i = 0; i < sizeof(numCoresArr) / sizeof(numCoresArr[0]); i++) {
        int numCores = numCoresArr[i];
        cout << "Running parallel BFS with " << numCores << " core(s)..." << endl;
        parallelBFS(graph, numCores);
    }

    return 0;
}
