# Message Passing between Processes

*January 14, 2014*

---

## Passing Data

Two main paradigms for sharing data between MPI processes:

- **One-to-all (broadcast):** one node sends the same data to all others → `MPI_Bcast`
- **All-to-one (reduce):** each node contributes data that is combined on one node → `MPI_Reduce`

### Example: Computing ln 2

$$\ln 2 = \sum_{n=1}^{N} \frac{(-1)^{n+1}}{n} = 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \cdots$$

Divide the terms among processors. Each processor computes a partial sum using only its share of the $n$ values, then the partial sums are reduced (summed) on the Boss node.

---

## Boss-Worker Model

- **Node 0** is designated the **Boss**.
- All other nodes are **Workers**.
- The Boss reads input data from a file and broadcasts it to all workers. Multiple processes cannot safely access the same file simultaneously.
- Results are collected (reduced) back to the Boss node.

---

## C++ Implementation: ln 2

```cpp
#include <iostream>
#include <fstream>
#include "mpi.h"

class MPI_stuff {
public:
    int NProcs;
    int MyID;
    MPI_stuff(int& argc, char**& argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &NProcs);
        MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
    }
    ~MPI_stuff() { MPI_Finalize(); }
};

double PartialSum(int N, int MyID, int NProcs)
{
    /* processor MyID handles terms n = MyID+1, MyID+1+NProcs, MyID+1+2*NProcs, ... */
    double sum = 0.0;
    for (int n = MyID + 1; n <= N; n += NProcs) {
        if (n % 2 == 1) sum += 1.0 / n;
        else            sum -= 1.0 / n;
    }
    return sum;
}

int main(int argc, char** argv)
{
    MPI_stuff the_mpi(argc, argv);

    int N = 0;
    if (the_mpi.MyID == 0) {          /* only Boss reads from file */
        std::ifstream fin("ln2data.txt");
        fin >> N;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);      /* send N to everyone */

    double mysum = PartialSum(N, the_mpi.MyID, the_mpi.NProcs);

    double Gsum = 0.0;
    MPI_Reduce(&mysum, &Gsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (the_mpi.MyID == 0)
        std::cout << "ln(2) ≈ " << Gsum << std::endl;

    return 0;
}
```

### Flowchart

![Boss-Worker flowchart for the ln 2 computation: Boss broadcasts N, all nodes compute partial sums, Workers reduce results to Boss.](ln2-flowchart.svg)

---

### MPI Function Signatures

**`MPI_Bcast`** — broadcast from root to all processes:

```c
MPI_Bcast(&N,     1, MPI_INT, 0, MPI_COMM_WORLD);
//         ↑      ↑    ↑      ↑       ↑
//       buffer count type  root  communicator
```

**`MPI_Reduce`** — combine values from all processes onto root:

```c
MPI_Reduce(&mysum, &Gsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//          ↑        ↑    ↑      ↑           ↑     ↑       ↑
//       send_buf recv_buf count type         op  root  communicator
```

---

## Arrays in MPI

If $N$ is not a multiple of `NProcs`, one simple fix is to zero-pad the end of the array.

### C-Style 1D Arrays

In C, arrays are laid out contiguously in memory:

![1D C array memory layout: elements a[0], a[1], … a[n-1] in adjacent cells.](array-layout.svg)

```c
float *a;
a = (float*) calloc(n, sizeof(float));   /* allocate n floats, zero-initialised */
```

> **Warning:** C arrays do **not** perform bounds checking. Accessing `a[-1]` or `a[n]` causes undefined behaviour with no error raised.

### MPI Data Types

| MPI type | C type |
|----------|--------|
| `MPI_INT` | `int` |
| `MPI_FLOAT` | `float` |
| `MPI_DOUBLE` | `double` |
| `MPI_CHAR` | `char` |

### Using `std::vector` (C++)

`std::vector` manages memory automatically and is safer than raw C arrays:

```cpp
std::vector<float> a(n);    /* allocate and zero-initialise n elements */
a.reserve(n);               /* reserve capacity without initialising */
a.resize(n);                /* set size, zero-initialising new elements */
```

---

## One-to-One Communication: Dot Product

$$\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i$$

Distribute the arrays across `NProcs` processors using a SIMD-style split: each processor receives a contiguous block of `size = n/NProcs` elements, computes a local dot product, then the partial sums are reduced (summed) to the Boss.

### Flowchart

![Boss-Worker flowchart for the dot product: Boss sends array chunks to Workers, all compute local dot products, results are reduced to Boss.](dotproduct-flowchart.svg)

---

### C++ Implementation: Dot Product

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include "mpi.h"

/* MPI_stuff class — same as above */

int GetNumberElements(std::ifstream& fin)
{
    int n;
    fin >> n;
    return n;
}

void ReadArrays(std::ifstream& fin, std::vector<float>& a,
                std::vector<float>& b, int n)
{
    for (int i = 0; i < n; i++)
        fin >> a[i] >> b[i];
}

float DotProduct(std::vector<float>& a, std::vector<float>& b,
                 int size, MPI_stuff& mpi)
{
    MPI_Status status;

    if (mpi.MyID == 0) {
        /* Boss sends each worker its chunk of both arrays */
        for (int i = 1; i < mpi.NProcs; i++) {
            MPI_Send(&a[size*i], size, MPI_FLOAT, i, 10, MPI_COMM_WORLD);
            MPI_Send(&b[size*i], size, MPI_FLOAT, i, 11, MPI_COMM_WORLD);
        }
    } else {
        /* Workers receive their chunks */
        MPI_Recv(&a[0], size, MPI_FLOAT, 0, 10, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], size, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, &status);
    }

    /* Each node computes its local dot product */
    float local_sum = 0.0f;
    for (int i = 0; i < size; i++)
        local_sum += a[i] * b[i];

    /* Reduce partial sums to Boss */
    float global_sum = 0.0f;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_sum;
}

/* See lecture notes for comments */
int main(int argc, char** argv)
{
    MPI_stuff the_mpi(argc, argv);

    std::ifstream fin;
    int n = 0;

    if (the_mpi.MyID == 0) {
        fin.open("DotData.txt");          /* only Boss reads from file */
        n = GetNumberElements(fin);       /* (multiple processes cannot access file simultaneously) */
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);   /* send n to everyone */

    if (n % the_mpi.NProcs) {            /* is n divisible by NProcs? */
        std::cout << "Number of processes is not a multiple of n.\n";
        MPI_Abort(MPI_COMM_WORLD, -2);   /* kills program gracefully on all nodes */
    }

    int size = n / the_mpi.NProcs;

    std::vector<float> a, b;
    if (the_mpi.MyID == 0) {
        a.reserve(n);  a.resize(n);      /* node 0 stores the full arrays */
        b.reserve(n);  b.resize(n);
        ReadArrays(fin, a, b, n);        /* node 0 reads from file */
    } else {
        a.reserve(size);  a.resize(size);  /* workers only need their chunk */
        b.reserve(size);  b.resize(size);
    }

    float adotb = DotProduct(a, b, size, the_mpi);   /* compute a·b */

    if (the_mpi.MyID == 0)
        std::cout << "The inner product is " << adotb << std::endl;

    return 0;
}
```

---

### `MPI_Send` and `MPI_Recv`

```c
MPI_Send(&a[size*i], size, MPI_FLOAT,  i,  10, MPI_COMM_WORLD);
//        ↑           ↑       ↑        ↑    ↑       ↑
//   ptr to buffer  count   type    dest  tag   communicator
//   (beginning of
//    buffer to send)            (processor
//                                to send to)

MPI_Recv(&a[0],      size, MPI_FLOAT,  0,  10, MPI_COMM_WORLD, &status);
//        ↑           ↑       ↑        ↑    ↑       ↑               ↑
//   ptr to recv   count   type    source  tag  communicator      status
//   buffer
```

> The **tag** in `MPI_Send` must match the tag in the corresponding `MPI_Recv`. *(These need to match — use a different tag for different messages.)*

---

### Design Comments

1. The `n % NProcs` condition can be fixed by zero-padding the end of the array.
2. The Boss uses more memory than the workers. This can be reduced by having the Boss read and send segments one at a time, then work on the last segment itself — which also resolves issue 1.
3. Potential crash points should call `MPI_Abort(MPI_COMM_WORLD, -2)` so that all MPI processes are stopped cleanly.

---

## Summary of MPI Commands Encountered So Far

```c
MPI_Init(&argc, &argv);
MPI_Finalize();
MPI_Abort(MPI_COMM_WORLD, -1);

MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

MPI_Bcast(&N,    1, MPI_INT,    0, MPI_COMM_WORLD);
MPI_Reduce(&sum, &Gsum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

MPI_Send(&a[size*i], size, MPI_FLOAT, i,  10, MPI_COMM_WORLD);
MPI_Recv(&a[0],      size, MPI_FLOAT, 0,  10, MPI_COMM_WORLD, &status);
```

---

## `MPI_Allreduce`

A variant of `MPI_Reduce` that distributes the result to **all** processors (All-to-All communication):

```c
MPI_Allreduce(&maxerr, &maxerrG, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
```

Here `maxerr` is reduced (e.g. finding the global maximum across all processors) and the result `maxerrG` is available on every processor — not only on one specific node.

### Reduction Operations

The `op` argument accepted by `MPI_Reduce`, `MPI_Allreduce`, `MPI_Reduce_scatter`, and `MPI_Scan`:

| Operation | Meaning |
|-----------|---------|
| `MPI_MAX` | maximum |
| `MPI_MIN` | minimum |
| `MPI_SUM` | sum |
| `MPI_PROD` | product |
| `MPI_LAND` | logical and |
| `MPI_BAND` | bit-wise and |
| `MPI_LOR` | logical or |
| `MPI_BOR` | bit-wise or |
| `MPI_LXOR` | logical xor |
| `MPI_BXOR` | bit-wise xor |
| `MPI_MAXLOC` | max value and location |
| `MPI_MINLOC` | min value and location |

*(Source: OpenMPI docs — https://www.open-mpi.org/doc/v3.0/man3/MPI_Reduce.3.php)*

---

## `MPI_Status`

`MPI_Status` is a data structure type with fields:

```c
MPI_Status status;
status.MPI_SOURCE   /* id of the processor that sent the message */
status.MPI_TAG      /* the message tag */
status.MPI_ERROR    /* error status (often not modified; same as function return value) */
/* other fields possible but implementation-dependent */
```

`status.MPI_SOURCE` is particularly useful when receiving with `MPI_ANY_SOURCE` — it tells you which processor the message actually came from.

---

## Controlling Synchronisation: Blocking vs Non-Blocking Communication

`MPI_Send` is a **blocking** routine: the sending processor waits for a "received" confirmation before working on the next line of code. Similarly, `MPI_Recv` causes the receiving processor to wait for the message before moving on.

---

## Example: Task Distribution

Suppose we have 10 tasks numbered 1 to 10 to distribute across `nprocs = 4` processors with $\mathit{nid} \in \{0, 1, 2, 3\}$.

```c
for (task=1; task <= 10; task++) {
    node = task - nprocs * (task/nprocs);   /* integer arithmetic — equivalent to task % nprocs */
    ...
}
```

| task | task/nprocs | node |
|------|-------------|------|
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 0 | 3 |
| 4 | 1 | 0 |
| 5 | 1 | 1 |
| 6 | 1 | 2 |
| 7 | 1 | 3 |
| 8 | 2 | 0 |
| 9 | 2 | 1 |
| 10 | 2 | 2 |

### Code (C with `mpi_vars` struct)

```c
#include <stdio.h>
#include "mpi.h"

/* mpi_vars struct and mpi_start() — see Ch1 refactored C version */

main(int argc, char** argv)
{
    struct mpi_vars this_mpi = mpi_start(argc, argv);
    MPI_Status status;
    int task, taskx, node;

    for (task=1; task <= 10; task++) {
        node = task - this_mpi.NProcs*(task/this_mpi.NProcs);  /* see which node does this task */

        if (node == this_mpi.MyID && node != 0) {    /* my task, and I'm not the Boss */
            if (node==2) waste_your_time();           /* some slow function */
            MPI_Send(&task, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);   /* tell Boss I am the one */
        }
        if (this_mpi.MyID == 0) {
            if (node == 0)
                printf("Processor %d will compute %d\n", 0, task);
            else {
                /* receive a message from anyone */
                MPI_Recv(&taskx, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status);
                printf("Processor %d will compute %d\n", status.MPI_SOURCE, taskx);
                /*                                       ↑ node the message actually came from */
            }
        }
    }

    for (task=11; task <= 20; task++) {
        node = task - this_mpi.NProcs*(task/this_mpi.NProcs);

        if (node == this_mpi.MyID && node != 0) {
            if (node==2) waste_your_time();
            MPI_Send(&task, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
        }
        if (this_mpi.MyID == 0) {
            if (node == 0)
                printf("Processor %d will compute %d\n", 0, task);
            else {
                MPI_Recv(&taskx, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status);
                printf("Processor %d will compute %d\n", status.MPI_SOURCE, taskx);
            }
        }
    }
    MPI_Finalize();
}
```

### Flowchart (1st Loop)

![Flowchart for the first task-distribution loop: two columns for nid=0 (Boss) and nid≠0 (Workers), showing the blocking MPI_Send/Recv pattern.](task-flowchart.svg)

### Sample Output (no synchronisation between loops)

```
Processor 1 will compute 1
Processor 1 will compute 5
Processor 1 will compute 9
Processor 1 will compute 13
Processor 1 will compute 17
Processor 3 will compute 3
Processor 3 will compute 7
Processor 3 will compute 11
Processor 3 will compute 15
Processor 3 will compute 19
Processor 0 will compute 4
Processor 0 will compute 8
Processor 0 will compute 12
Processor 2 will compute 2
Processor 2 will compute 6
Processor 0 will compute 16
Processor 2 will compute 10
Processor 2 will compute 14
Processor 2 will compute 18
Processor 0 will compute 20
```

**Comments:**
1. Tasks are not completed in order (and the order may change on each run).
2. Node 2 is slow, as expected, because of the `waste_your_time()` call.
3. Node 0 is slower because it is busy receiving results from other processors and outputting.
4. Nodes 1 and 3 finish all 20 tasks before nodes 0 and 2 finish even their first — they complete both loops before nodes 0 and 2 get through the first.

---

## `MPI_Barrier`

```c
MPI_Barrier(MPI_COMM_WORLD);
```

Blocks the calling process from proceeding until **all** processes in `MPI_COMM_WORLD` have made a similar call. Adding `MPI_Barrier` between the two task loops ensures all tasks 1–10 are complete before tasks 11–20 begin.

### Sample Output (with `MPI_Barrier` between loops)

```
Processor 1 will compute 1
Processor 3 will compute 3
Processor 1 will compute 5
Processor 3 will compute 7
Processor 1 will compute 9
Processor 0 will compute 4
Processor 2 will compute 2
Processor 0 will compute 8
Processor 2 will compute 6
Processor 2 will compute 10
Processor 1 will compute 13
Processor 3 will compute 11
Processor 1 will compute 17
Processor 3 will compute 15
Processor 3 will compute 19
Processor 0 will compute 12
Processor 0 will compute 16
Processor 2 will compute 14
Processor 2 will compute 18
Processor 0 will compute 20
```

---

## Ordering with Blocking `MPI_Send`

Since `MPI_Send` is blocking, you can force tasks to arrive in order by using the task number as the tag:

```c
MPI_Send(&task, 1, MPI_INT, 0, 10+node, MPI_COMM_WORLD);  /* tag = 10+node */
...
MPI_Recv(&taskx, 1, MPI_INT, MPI_ANY_SOURCE, 10+node, MPI_COMM_WORLD, &status);
```

> **Note:** This will slow things down significantly — a processor that finishes early must now wait for node 0 to receive the previous task before it can proceed. Effectively, tasks are executed sequentially in time, just on different processors, which eliminates the benefit of parallelism.

---

## Non-Blocking Communication

`MPI_Isend` and `MPI_Irecv` are **non-blocking** variants: the calling processor submits the communication request and immediately proceeds to the next line of code without waiting for delivery confirmation.

```c
MPI_Request req_send10 = MPI_REQUEST_NULL;
MPI_Request req_recv10 = MPI_REQUEST_NULL;

MPI_Isend(&old[size-1][1], N-1, MPI_FLOAT, nid+1, 10, MPI_COMM_WORLD, &req_send10);
MPI_Irecv(&old[0][1],      N-1, MPI_FLOAT, nid+1, 10, MPI_COMM_WORLD, &req_recv10);
MPI_Wait(&req_recv10, &status);
```

- `MPI_REQUEST_NULL` is the default initialisation value for any `MPI_Request` object.
- `MPI_Wait` blocks until the operation associated with `req_recv10` has completed successfully.
- Send requests often do not need confirmation; receive requests do. Note that `MPI_Irecv` + `MPI_Wait` is effectively equivalent to `MPI_Recv`.

### Revised Task Loop (Non-Blocking `MPI_Isend`)

```c
for (task=1; task <= 10; task++) {
    node = task - this_mpi.NProcs*(task/this_mpi.NProcs);

    if (node == this_mpi.MyID) {
        if (node==2) waste_your_time();
        MPI_Isend(&task, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &req_send10);
        /* req_send10 previously declared of type MPI_Request */
    }
    if (this_mpi.MyID == 0) {
        req_recv10 = MPI_REQUEST_NULL;   /* must set now so not previously passed */
        MPI_Irecv(&taskx, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &req_recv10);
        MPI_Wait(&req_recv10, &status);
        printf("Processor %d will compute %d\n", status.MPI_SOURCE, taskx);
    }
}
```

This removes the blocking effect of `MPI_Send` and allows all processes (including `MyID == 0`) to proceed through the full task list without waiting for the Boss to confirm receipt of each message.

### Separating Send and Receive Loops (Boss Avoids Waiting)

To avoid having the Boss wait for others to complete before doing its own task, move the receives into a separate loop:

```c
for (task=1; task <= 10; task++) {
    node = task - this_mpi.NProcs*(task/this_mpi.NProcs);

    if (node == this_mpi.MyID) {
        if (node==2) waste_your_time();
        MPI_Isend(&task, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &req_send10);
    }
}

if (this_mpi.MyID == 0) {
    for (task=1; task <= 10; task++) {
        req_recv10 = MPI_REQUEST_NULL;
        MPI_Irecv(&taskx, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &req_recv10);
        MPI_Wait(&req_recv10, &status);
        printf("Processor %d will compute %d\n", status.MPI_SOURCE, taskx);
    }
}
MPI_Barrier(MPI_COMM_WORLD);
/* A good idea to put MPI_Barrier here to prevent more messages being sent to nid=0
   by other processors and getting confused with these tasks, since MPI_Irecv is
   very open (non-specific — it accepts messages from anyone). */
```
