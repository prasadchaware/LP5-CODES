#include <stdio.h>
#include <omp.h>

// Sequential Bubble Sort
void bubbleSort(int arr[], int n) {
    int i, j, temp;
    for (i = 0; i < n-1; i++) {
        for (j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

// Parallel Bubble Sort using OpenMP
void parallelBubbleSort(int arr[], int n) {
    int i, j, temp;
    #pragma omp parallel for private(j, temp)
    for (i = 0; i < n-1; i++) {
        for (j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

// Sequential Merge Sort
void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
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

void sequentialMergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort using OpenMP
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

int main() {
    int arr[] = {12, 11, 13, 5, 6, 7};
    int n = sizeof(arr)/sizeof(arr[0]);

    // Perform sequential bubble sort
    bubbleSort(arr, n);

    // Perform parallel bubble sort
    parallelBubbleSort(arr, n);

    // Perform sequential merge sort
    sequentialMergeSort(arr, 0, n - 1);

    // Perform parallel merge sort
    parallelMergeSort(arr, 0, n - 1);

    return 0;
}
