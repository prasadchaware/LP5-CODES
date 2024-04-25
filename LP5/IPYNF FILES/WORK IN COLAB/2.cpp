#include <iostream>
#include <omp.h>

using namespace std;

// Parallel Bubble Sort
void parallelBubbleSort(int arr[], int n) {
    bool swapped = true;
    #pragma omp parallel
    {
        while (swapped) {
            swapped = false;
            #pragma omp for
            for (int i = 0; i < n-1; i++) {
                if (arr[i] > arr[i+1]) {
                    swap(arr[i], arr[i+1]);
                    swapped = true;
                }
            }
        }
    }
}


// Merge function for Merge Sort
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Parallel Merge Sort
void parallelMergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);
            #pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

// Function to print an array
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        cout << arr[i] << " ";
    cout << endl;
}

int main() {
    int arr[] = {64, 25, 12, 22, 11};
    int n = sizeof(arr) / sizeof(arr[0]);

    cout << "Original array: \n";
    printArray(arr, n);

    // Parallel Bubble Sort
    parallelBubbleSort(arr, n);
    cout << "Array sorted by parallel Bubble Sort: \n";
    printArray(arr, n);

    int arr2[] = {12, 11, 13, 5, 6, 7};
    int n2 = sizeof(arr2) / sizeof(arr2[0]);

    cout << "\nOriginal array: \n";
    printArray(arr2, n2);

    // Parallel Merge Sort
    parallelMergeSort(arr2, 0, n2 - 1);
    cout << "Array sorted by parallel Merge Sort: \n";
    printArray(arr2, n2);

    return 0;
}
