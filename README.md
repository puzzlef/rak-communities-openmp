Design of OpenMP-based Parallel [Label Propagation Algorithm (LPA)], aka RAK, for [community detection].

In recent years, there has been an unprecedented increase in data collection, and their representation as graphs. Thus, there is a demand for efficient parallel algorithms designed to identify communities in large networks. The significance of the multicore/shared memory environment in this context is crucial, both due to its energy efficiency, and due to widespread availability of hardware with large memory capacities. Existing research on Label Propagation Algorithm (LPA) focuses on algorithmic optimizations and parallelization but lacks exploration of efficient data structures for selecting the most weighted label. Furthermore, the proposed techniques are dispersed across multiple papers, making it challenging for readers to grasp and implement them effectively. To address this, we introduce **GVE-LPA**, an optimized *parallel implementation of LPA* for shared memory multicores.

Below we plot the time taken by [FLPA], [igraph] LPA, [NetworKit] LPA, and GVE-LPA on 13 different graphs. GVE-LPA surpasses FLPA, igraph LPA, and NetworKit LPA by `139×`, `97,000×`, and `40×` respectively, achieving a processing rate of `1.4𝐵` edges/s on a `3.8𝐵` edge graph.

[![](https://i.imgur.com/dWFUkZG.png)][sheets-o1]

Below we plot the speedup of GVE-LPA wrt FLPA, igraph LPA, and NetworKit LPA.

[![](https://i.imgur.com/buUUyke.png)][sheets-o1]

Next, we plot the modularity of communities identified by FLPA, igraph LPA, NetworKit LPA, and GVE-LPA. GVE-LPA on average obtains `6.6%` / `0.2%` higher modularity than FLPA and igraph LPA respectively (especially on web graphs), and `4.1%` lower modularity than NetworKit LPA (especially on protein k-mer graphs with large number of
vertices and a low average degree).

[![](https://i.imgur.com/AQzlelS.png)][sheets-o1]

Finally, we plot the strong scaling behaviour of GVE-LPA. With doubling of threads, GVE-LPA exhibits an average performance scaling of `1.7×`.

[![](https://i.imgur.com/yUx3fNy.png)][sheets-o2]

Refer to our technical report for more details: \
[GVE-LPA: Fast Label Propagation Algorithm (LPA) for Community Detection in Shared Memory Setting][report].

<br>

> [!NOTE]
> You can just copy `main.sh` to your system and run it. \
> For the code, refer to `main.cxx`.


[Label Propagation Algorithm (LPA)]: https://arxiv.org/abs/0709.2938
[FLPA]: https://github.com/vtraag/igraph/tree/flpa
[igraph]: https://github.com/igraph/igraph
[NetworKit]: https://github.com/networkit/networkit
[community detection]: https://en.wikipedia.org/wiki/Community_search
[sheets-o1]: https://docs.google.com/spreadsheets/d/1JQ7wpFC0qgi_isdaPz0OusSRHXONiKKQeG9JmTm3J4U/edit?usp=sharing
[sheets-o2]: https://docs.google.com/spreadsheets/d/1fLPk0cxFYVCTz-LXPq1HyyP0yVOwmI5ry5KWwFNQIH0/edit?usp=sharing
[report]: https://arxiv.org/abs/2312.08140

<br>
<br>


### Code structure

The code structure of GVE-LPA is as follows:

```bash
- inc/_algorithm.hxx: Algorithm utility functions
- inc/_bitset.hxx: Bitset manipulation functions
- inc/_cmath.hxx: Math functions
- inc/_ctypes.hxx: Data type utility functions
- inc/_cuda.hxx: CUDA utility functions
- inc/_debug.hxx: Debugging macros (LOG, ASSERT, ...)
- inc/_iostream.hxx: Input/output stream functions
- inc/_iterator.hxx: Iterator utility functions
- inc/_main.hxx: Main program header
- inc/_mpi.hxx: MPI (Message Passing Interface) utility functions
- inc/_openmp.hxx: OpenMP utility functions
- inc/_queue.hxx: Queue utility functions
- inc/_random.hxx: Random number generation functions
- inc/_string.hxx: String utility functions
- inc/_utility.hxx: Runtime measurement functions
- inc/_vector.hxx: Vector utility functions
- inc/batch.hxx: Batch update generation functions
- inc/bfs.hxx: Breadth-first search algorithms
- inc/csr.hxx: Compressed Sparse Row (CSR) data structure functions
- inc/dfs.hxx: Depth-first search algorithms
- inc/duplicate.hxx: Graph duplicating functions
- inc/Graph.hxx: Graph data structure functions
- inc/rak.hxx: LPA/RAK community detection algorithm functions
- inc/main.hxx: Main header
- inc/mtx.hxx: Graph file reading functions
- inc/properties.hxx: Graph Property functions
- inc/selfLoop.hxx: Graph Self-looping functions
- inc/symmetricize.hxx: Graph Symmetricization functions
- inc/transpose.hxx: Graph transpose functions
- inc/update.hxx: Update functions
- main.cxx: Experimentation code
- process.js: Node.js script for processing output logs
```

Note that each branch in this repository contains code for a specific experiment. The `main` branch contains code for the final experiment. If the intention of a branch in unclear, or if you have comments on our technical report, feel free to open an issue.

<br>
<br>


## References

- [Near linear time algorithm to detect community structures in large-scale networks; Raghavan et al. (2007)](https://arxiv.org/abs/0709.2938)
- [The University of Florida Sparse Matrix Collection; Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)
- [How to import VSCode keybindings into Visual Studio?](https://stackoverflow.com/a/62417446/1413259)
- [Configure X11 Forwarding with PuTTY and Xming](https://www.centlinux.com/2019/01/configure-x11-forwarding-putty-xming-windows.html)
- [Installing snap on CentOS](https://snapcraft.io/docs/installing-snap-on-centos)

<br>
<br>


[![](https://i.imgur.com/7QLfaW3.jpg)](https://www.youtube.com/watch?v=IwiYQILYXDQ)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
![](https://ga-beacon.deno.dev/G-KD28SG54JQ:hbAybl6nQFOtmVxW4if3xw/github.com/puzzlef/rak-communities-openmp)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
