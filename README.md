# Distributed Systems Failure & Consistency Simulators

Interactive Python simulations to understand how distributed systems fail and how to prevent those failures.

## Simulators

### 1. Metastable Failure Simulator (`metastable_simulator.py`)
Watch retry storms create self-sustaining system collapse. You'll see how a small spike in traffic can trigger a cascade that persists even after the spike ends.

**You'll learn**: Why backoff matters, how coordination overhead kills systems, when circuit breakers save you.

### 2. CAP/PACELC Consistency Simulator (`cap_consistency_simulator.py`)
Compare quorum-based consistency (R+W>N) against eventual consistency. See the real latency costs of coordination.

**You'll learn**: When to choose consistency over latency, how network partitions affect availability, why most systems use eventual consistency.

### 3. CRDT Simulator (`crdt_simulator.py`)
Watch CRDTs converge without any coordination. See how mathematical properties guarantee eventual consistency.

**You'll learn**: G-Counters, PN-Counters, OR-Sets, LWW-Maps, when CRDTs work (and when they don't).

### 4. Cross-System Interaction Failure Simulator (`csi_failure_simulator.py`)
See how two working systems fail when connected. Based on research showing 20% of cloud incidents are CSI failures.

**You'll learn**: Why schema mismatches crash systems, how config incoherence causes failures, why testing in isolation isn't enough.

## Quick Start

### Install Dependencies
```bash
git clone https://github.com/bhatti/simulators
cd simulators
pip install -r requirements.txt
```

### Run Simulators

**Option 1: Interactive Menu**
```bash
python run_all_simulators.py
```

**Option 2: Run Individual Simulators**
```bash
streamlit run metastable_simulator.py
streamlit run cap_consistency_simulator.py
streamlit run crdt_simulator.py
streamlit run csi_failure_simulator.py
```

The simulators will open in your browser with interactive controls, real-time charts, and educational content.

## How to Use

Each simulator has:
- **Sliders**: Adjust system parameters
- **Run Button**: Start the simulation
- **Charts**: Watch failures happen in real-time
- **Expandable Sections**: Learn the theory behind what you're seeing
- **Experiment Ideas**: Guided explorations to build intuition

## Example: Seeing Metastability

1. Run `streamlit run metastable_simulator.py`
2. Set load pattern to "spike"
3. Disable "Enable Exponential Backoff"
4. Click "Run Simulation"
5. Watch P99 latency shoot up and stay high even after the spike ends
6. Enable backoff and run again - see the difference!

## What You'll Learn

### Metastable Failures
- Why small traffic spikes cause permanent degradation
- How retry storms amplify problems
- Why your retry logic might be making things worse
- When to use circuit breakers vs backoff

### Consistency Tradeoffs  
- Real latency costs of strong consistency (3x slower!)
- When eventual consistency is good enough
- How network partitions force you to choose availability or consistency
- Why most systems pick latency over consistency in normal operation

### CRDTs
- How to get consistency without coordination
- Why order doesn't matter for some operations
- When you can avoid distributed locks
- Where CRDTs break down (spoiler: constraints and uniqueness)

### Cross-System Failures
- Why 20% of cloud incidents involve working systems failing together
- How schema mismatches cause crashes
- Why configuration is a cross-system problem
- How to test systems together, not just alone

## Technical Foundation

Built with:
- **SimPy**: Discrete event simulation
- **Streamlit**: Interactive web UI
- **Plotly**: Real-time charts
- **NumPy/Pandas**: Statistical analysis

## Based On

**Research Papers**:
- [AWS HotOS 2025: Metastability prediction](https://sigops.org/s/conferences/hotos/2025/papers/hotos25-106.pdf)
- [Abadi 2012: PACELC framework](https://www.cs.umd.edu/%7Eabadi/papers/abadi-pacelc.pdf)
- [Shapiro et al. 2011: CRDTs](https://inria.hal.science/hal-00932836/file/CRDTs_SSS-2011.pdf)
- [EuroSys 2023: Cross-system interaction failures](http://dprg.cs.uiuc.edu/data/files/2023/eurosys23-fall-final-CSI.pdf)

**Industry Blogs**:
- [Marc Brooker (AWS): Aurora DSQL design](https://brooker.co.za/blog/2025/11/02/thinking-dsql.html)
- [James Hamilton: Operations-friendly design](https://mvdirona.com/jrh/talksAndPapers/JamesRH_Lisa.pdf)

## License

MIT License - Use freely for education and research.

