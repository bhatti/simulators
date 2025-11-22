"""
CAP/PACELC Consistency Simulator
Demonstrates consistency vs availability tradeoffs in distributed databases
Shows quorum-based systems vs eventual consistency approaches
"""

import streamlit as st
import simpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ConsistencyLevel(Enum):
    """Different consistency models"""
    STRONG = "Strong (Quorum: R+W>N)"
    EVENTUAL = "Eventual Consistency"
    LINEARIZABLE = "Linearizable (R=W=N)"
    WEAK = "Weak (R=1, W=1)"

@dataclass
class WriteOperation:
    """A write operation with version"""
    key: str
    value: int
    version: int
    timestamp: float
    origin_node: int

@dataclass
class ReadOperation:
    """A read operation"""
    key: str
    request_id: int
    timestamp: float

class ReplicaNode:
    """A single replica node in the distributed system"""
    def __init__(self, env, node_id, base_latency, network_latency, failure_prob=0.0):
        self.env = env
        self.node_id = node_id
        self.base_latency = base_latency
        self.network_latency = network_latency
        self.failure_prob = failure_prob
        self.is_failed = False
        
        # Data store: key -> (value, version, timestamp)
        self.data = {}
        
        # Pending replication queue
        self.replication_queue = []
        
    def write_local(self, write_op: WriteOperation):
        """Write to local node"""
        yield self.env.timeout(self.base_latency)
        
        if random.random() < self.failure_prob:
            self.is_failed = True
            return False
        
        # Check version for conflict
        current = self.data.get(write_op.key)
        if current is None or write_op.version > current[1]:
            self.data[write_op.key] = (write_op.value, write_op.version, write_op.timestamp)
        
        return True
    
    def read_local(self, key: str):
        """Read from local node"""
        yield self.env.timeout(self.base_latency)
        
        if random.random() < self.failure_prob:
            self.is_failed = True
            return None
        
        return self.data.get(key)
    
    def replicate_to(self, target_node, write_op: WriteOperation):
        """Asynchronously replicate to another node"""
        # Network delay
        yield self.env.timeout(self.network_latency)
        
        if not target_node.is_failed:
            yield self.env.process(target_node.write_local(write_op))

class DistributedDatabase:
    """Distributed database with configurable consistency"""
    def __init__(self, env, n_nodes, base_latency, network_latency, 
                 consistency_level, partition_active=False):
        self.env = env
        self.n_nodes = n_nodes
        self.consistency_level = consistency_level
        self.partition_active = partition_active
        
        # Create replica nodes
        self.nodes = [
            ReplicaNode(env, i, base_latency, network_latency)
            for i in range(n_nodes)
        ]
        
        # Version counter for writes
        self.version_counter = 0
        
        # Stats
        self.stats = {
            'write_latencies': [],
            'read_latencies': [],
            'inconsistent_reads': 0,
            'failed_writes': 0,
            'failed_reads': 0,
            'total_writes': 0,
            'total_reads': 0,
            'timestamps': []
        }
    
    def get_quorum_size(self, operation_type):
        """Calculate quorum size based on consistency level"""
        if self.consistency_level == ConsistencyLevel.STRONG:
            # R+W > N, use majority for both
            return (self.n_nodes // 2) + 1
        elif self.consistency_level == ConsistencyLevel.LINEARIZABLE:
            # R=W=N
            return self.n_nodes
        elif self.consistency_level == ConsistencyLevel.WEAK:
            # Weak: R=1, W=1
            return 1
        else:  # EVENTUAL
            # Eventual: W=1, async replication
            return 1
    
    def simulate_partition(self):
        """Simulate network partition"""
        if self.partition_active:
            # Partition: nodes 0,1 can't reach node 2
            partition_size = self.n_nodes // 2
            for i in range(partition_size):
                self.nodes[i].is_failed = False
            for i in range(partition_size, self.n_nodes):
                self.nodes[i].is_failed = True
    
    def write(self, key: str, value: int, client_id: int):
        """Perform a write operation"""
        start_time = self.env.now
        self.stats['total_writes'] += 1
        self.version_counter += 1
        
        write_op = WriteOperation(
            key=key,
            value=value,
            version=self.version_counter,
            timestamp=start_time,
            origin_node=0
        )
        
        # Apply partition if active
        self.simulate_partition()
        
        # Get available nodes
        available_nodes = [n for n in self.nodes if not n.is_failed]
        quorum_size = self.get_quorum_size('write')
        
        if len(available_nodes) < quorum_size:
            # Cannot meet quorum - write fails (AP system would succeed)
            self.stats['failed_writes'] += 1
            if self.consistency_level in [ConsistencyLevel.STRONG, ConsistencyLevel.LINEARIZABLE]:
                # CP system: fail write
                yield self.env.timeout(0.1)
                self.stats['write_latencies'].append(self.env.now - start_time)
                return False
            else:
                # AP system: write to available nodes
                available_nodes = [available_nodes[0]] if available_nodes else []
        
        # Write to quorum nodes
        write_results = []
        for node in available_nodes[:quorum_size]:
            result = yield self.env.process(node.write_local(write_op))
            write_results.append(result)
        
        # Async replication for eventual consistency
        if self.consistency_level == ConsistencyLevel.EVENTUAL:
            for node in self.nodes[quorum_size:]:
                if not node.is_failed:
                    self.env.process(node.replicate_to(node, write_op))
        
        success = sum(write_results) >= quorum_size
        latency = self.env.now - start_time
        self.stats['write_latencies'].append(latency)
        self.stats['timestamps'].append(self.env.now)
        
        return success
    
    def read(self, key: str, expected_value: Optional[int] = None):
        """Perform a read operation"""
        start_time = self.env.now
        self.stats['total_reads'] += 1
        
        # Apply partition if active
        self.simulate_partition()
        
        # Get available nodes
        available_nodes = [n for n in self.nodes if not n.is_failed]
        quorum_size = self.get_quorum_size('read')
        
        if len(available_nodes) < quorum_size:
            self.stats['failed_reads'] += 1
            yield self.env.timeout(0.1)
            self.stats['read_latencies'].append(self.env.now - start_time)
            return None
        
        # Read from quorum nodes
        read_results = []
        for node in available_nodes[:quorum_size]:
            result = yield self.env.process(node.read_local(key))
            read_results.append(result)
        
        # Get most recent version (read repair logic)
        valid_results = [r for r in read_results if r is not None]
        
        if not valid_results:
            self.stats['failed_reads'] += 1
            latency = self.env.now - start_time
            self.stats['read_latencies'].append(latency)
            return None
        
        # Return highest version
        result = max(valid_results, key=lambda x: x[1])  # Sort by version
        
        # Check for consistency
        if expected_value is not None and result[0] != expected_value:
            self.stats['inconsistent_reads'] += 1
        
        latency = self.env.now - start_time
        self.stats['read_latencies'].append(latency)
        self.stats['timestamps'].append(self.env.now)
        
        return result[0]

class Client:
    """Client generating read/write workload"""
    def __init__(self, env, db, client_id, write_ratio=0.3, rate=5):
        self.env = env
        self.db = db
        self.client_id = client_id
        self.write_ratio = write_ratio
        self.rate = rate
        self.last_written = {}  # Track what we wrote for consistency checks
        
    def run(self):
        """Generate workload"""
        while True:
            yield self.env.timeout(random.expovariate(self.rate))
            
            key = f"key_{random.randint(0, 10)}"
            
            if random.random() < self.write_ratio:
                # Write operation
                value = random.randint(1, 1000)
                success = yield self.env.process(self.db.write(key, value, self.client_id))
                if success:
                    self.last_written[key] = value
            else:
                # Read operation - check if we get what we last wrote
                expected = self.last_written.get(key)
                result = yield self.env.process(self.db.read(key, expected))

def run_simulation(consistency_level, n_nodes, n_clients, duration,
                   write_ratio, base_latency, network_latency, partition_active):
    """Run the distributed database simulation"""
    
    env = simpy.Environment()
    
    db = DistributedDatabase(
        env, n_nodes, base_latency, network_latency,
        consistency_level, partition_active
    )
    
    # Create clients
    clients = [
        Client(env, db, i, write_ratio, rate=5)
        for i in range(n_clients)
    ]
    
    # Start clients
    for client in clients:
        env.process(client.run())
    
    # Run simulation
    env.run(until=duration)
    
    return db.stats

def plot_consistency_results(stats, title, consistency_level):
    """Visualize consistency simulation results"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Read vs Write Latency Distribution',
            'Latency Over Time',
            'Consistency Metrics',
            'Success Rates'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"type": "bar"}, {"type": "indicator"}]
        ]
    )
    
    # 1. Latency distributions
    if stats['write_latencies']:
        fig.add_trace(
            go.Box(y=stats['write_latencies'], name='Write Latency',
                   marker_color='red', boxmean=True),
            row=1, col=1
        )
    if stats['read_latencies']:
        fig.add_trace(
            go.Box(y=stats['read_latencies'], name='Read Latency',
                   marker_color='blue', boxmean=True),
            row=1, col=1
        )
    
    # 2. Latency over time
    if stats['timestamps']:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(stats['write_latencies']))),
                y=stats['write_latencies'],
                mode='markers',
                name='Write',
                marker=dict(color='red', size=3, opacity=0.5)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(len(stats['read_latencies']))),
                y=stats['read_latencies'],
                mode='markers',
                name='Read',
                marker=dict(color='blue', size=3, opacity=0.5)
            ),
            row=1, col=2
        )
    
    # 3. Consistency metrics
    metrics = [
        'Inconsistent Reads',
        'Failed Writes',
        'Failed Reads'
    ]
    values = [
        stats['inconsistent_reads'],
        stats['failed_writes'],
        stats['failed_reads']
    ]
    colors = ['orange', 'red', 'darkred']
    
    fig.add_trace(
        go.Bar(x=metrics, y=values, marker_color=colors,
               text=values, textposition='auto'),
        row=2, col=1
    )
    
    # 4. Overall success rate
    total_ops = stats['total_reads'] + stats['total_writes']
    failed_ops = stats['failed_reads'] + stats['failed_writes']
    success_rate = ((total_ops - failed_ops) / total_ops * 100) if total_ops > 0 else 0
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=success_rate,
            title={'text': "Success Rate"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 90], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text=title,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Latency (s)", row=1, col=1)
    fig.update_xaxes(title_text="Operation #", row=1, col=2)
    fig.update_yaxes(title_text="Latency (s)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    return fig

# Streamlit UI
st.set_page_config(layout="wide", page_title="CAP/PACELC Simulator")

st.title("⚖️ CAP Theorem & Consistency Simulator")

st.markdown("""
Explore the tradeoffs between **Consistency**, **Availability**, and **Partition Tolerance** (CAP),
and understand how **Latency** vs **Consistency** plays out in normal operation (PACELC).

**Key Concepts**:
- **Strong Consistency (Quorum)**: R+W>N guarantees linearizability but requires coordination
- **Eventual Consistency**: High availability & low latency, eventual convergence
- **During Partitions**: CP systems sacrifice availability, AP systems sacrifice consistency
""")

# Controls
col1, col2 = st.columns(2)

with col1:
    st.subheader("🗄️ System Configuration")
    
    consistency_level = st.selectbox(
        "Consistency Model",
        [ConsistencyLevel.STRONG, ConsistencyLevel.EVENTUAL, 
         ConsistencyLevel.LINEARIZABLE, ConsistencyLevel.WEAK],
        format_func=lambda x: x.value,
        help="Strong: R+W>N, Eventual: async replication, Linearizable: R=W=N"
    )
    
    n_nodes = st.slider("Number of Replica Nodes", 3, 9, 5, step=2,
                       help="Odd numbers recommended for quorum")
    
    base_latency = st.slider("Node Processing Latency (ms)", 1, 50, 10)
    network_latency = st.slider("Network Latency (ms)", 10, 200, 50,
                                help="WAN: ~50-150ms, Local: ~1-10ms")

with col2:
    st.subheader("👥 Workload Configuration")
    
    n_clients = st.slider("Number of Clients", 1, 20, 5)
    write_ratio = st.slider("Write Ratio", 0.0, 1.0, 0.3, 0.1,
                           help="Fraction of operations that are writes")
    
    partition_active = st.checkbox(
        "Simulate Network Partition",
        help="Partition the cluster to test CAP behavior"
    )
    
    duration = st.slider("Simulation Duration (s)", 10, 100, 30)

# Classification helper
def classify_system(consistency_level, partition_active):
    """Classify system according to PACELC"""
    if partition_active:
        if consistency_level in [ConsistencyLevel.STRONG, ConsistencyLevel.LINEARIZABLE]:
            return "**PC**: Partition → Choose Consistency (sacrifice Availability)"
        else:
            return "**PA**: Partition → Choose Availability (sacrifice Consistency)"
    else:
        if consistency_level in [ConsistencyLevel.STRONG, ConsistencyLevel.LINEARIZABLE]:
            return "**EC**: Normal → Choose Consistency (higher Latency)"
        else:
            return "**EL**: Normal → Choose Latency (eventual Consistency)"

classification = classify_system(consistency_level, partition_active)
st.info(f"**System Classification (PACELC)**: {classification}")

# Run simulation
if st.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Running distributed system simulation..."):
        stats = run_simulation(
            consistency_level, n_nodes, n_clients, duration,
            write_ratio, base_latency/1000, network_latency/1000,
            partition_active
        )
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reads", stats['total_reads'])
        with col2:
            st.metric("Total Writes", stats['total_writes'])
        with col3:
            st.metric("Inconsistent Reads", stats['inconsistent_reads'],
                     delta=None if stats['inconsistent_reads'] == 0 else "⚠️")
        with col4:
            failed = stats['failed_reads'] + stats['failed_writes']
            st.metric("Failed Operations", failed,
                     delta=None if failed == 0 else "❌")
        
        # Latency metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if stats['write_latencies']:
            with col1:
                st.metric("Avg Write Latency", 
                         f"{np.mean(stats['write_latencies'])*1000:.1f} ms")
            with col2:
                st.metric("P99 Write Latency",
                         f"{np.percentile(stats['write_latencies'], 99)*1000:.1f} ms")
        
        if stats['read_latencies']:
            with col3:
                st.metric("Avg Read Latency",
                         f"{np.mean(stats['read_latencies'])*1000:.1f} ms")
            with col4:
                st.metric("P99 Read Latency",
                         f"{np.percentile(stats['read_latencies'], 99)*1000:.1f} ms")
        
        # Plot
        fig = plot_consistency_results(
            stats, 
            f"CAP Analysis: {consistency_level.value}",
            consistency_level
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis
        st.subheader("📊 Analysis")
        
        # Latency analysis
        if stats['read_latencies'] and stats['write_latencies']:
            avg_read = np.mean(stats['read_latencies']) * 1000
            avg_write = np.mean(stats['write_latencies']) * 1000
            latency_ratio = avg_read / avg_write if avg_write > 0 else 1.0
            
            if consistency_level in [ConsistencyLevel.STRONG, ConsistencyLevel.LINEARIZABLE]:
                expected_ratio = "~1x (both require quorum)"
            else:
                expected_ratio = "~1x (single node)"
            
            st.write(f"""
            **Latency Characteristics**:
            - Read/Write Latency Ratio: {latency_ratio:.2f}x
            - Expected for {consistency_level.value}: {expected_ratio}
            """)
        
        # Consistency analysis
        if stats['inconsistent_reads'] > 0:
            st.warning(f"""
            ⚠️ **Consistency Violations Detected**
            
            Found {stats['inconsistent_reads']} inconsistent reads where clients read stale data.
            This is expected for:
            - Eventual consistency models (data is still propagating)
            - During network partitions in AP systems
            
            Strong consistency with R+W>N should prevent this in normal operation.
            """)
        
        # Availability analysis
        if partition_active:
            if stats['failed_writes'] > 0 or stats['failed_reads'] > 0:
                st.error(f"""
                ❌ **Availability Impact During Partition**
                
                With {n_nodes} nodes and a partition, some operations failed:
                - Failed reads: {stats['failed_reads']}
                - Failed writes: {stats['failed_writes']}
                
                This is expected for **CP systems** (Consistency over Availability).
                
                **Minority partition**: Cannot form quorum → Operations fail
                **Majority partition**: Can still serve requests
                
                For **AP systems**: Would continue serving requests with stale data.
                """)
            else:
                st.success("✅ All operations succeeded despite partition (AP system)")

# Educational sections
with st.expander("📚 Understanding CAP and PACELC"):
    st.markdown("""
    ### The CAP Theorem (1998)
    
    **States**: In a distributed system, you can only guarantee 2 of 3:
    - **C**onsistency: All nodes see the same data at the same time
    - **A**vailability: Every request receives a response
    - **P**artition Tolerance: System continues despite network failures
    
    **Reality**: You must choose P (partitions will happen), so you choose between C or A *during partitions*.
    
    ### The PACELC Framework (2012)
    
    **More Complete Picture**:
    - **If Partition (P)**: Choose between Availability (A) and Consistency (C)
    - **Else (E)**: Choose between Latency (L) and Consistency (C)
    
    **Key Insight**: Most design decisions are driven by the Latency/Consistency tradeoff 
    during *normal operation*, not just partition behavior!
    
    ### System Classifications
    
    | System | Partition (P) | Normal (E) | Examples |
    |--------|--------------|------------|----------|
    | **PA/EL** | A over C | L over C | Dynamo, Cassandra, Riak |
    | **PC/EC** | C over A | C over L | Spanner, VoltDB, HBase |
    | **PA/EC** | A over C | C over L | MongoDB |
    | **PC/EL** | C over A | L over C | PNUTS |
    
    ### Quorum Systems (R+W>N)
    
    **How it works**:
    - Write to W nodes
    - Read from R nodes
    - If R+W > N, read and write sets overlap → consistency guarantee
    
    **Example** (N=5, R=3, W=3):
    - Write set: {1, 2, 3}
    - Read set: {2, 3, 4}
    - Overlap: {2, 3} ensures read sees at least one up-to-date replica
    
    **Cost**: Every operation requires network round-trips to multiple nodes
    
    ### Eventual Consistency (CRDTs)
    
    **How it works**:
    - Updates can be applied independently
    - No coordination required
    - Guaranteed convergence through commutative operations
    
    **Cost**: Temporary inconsistency, more complex conflict resolution
    
    **Best for**: High availability, low latency, partition tolerance
    """)

with st.expander("🎯 Real-World Tradeoffs"):
    st.markdown("""
    ### When to Choose Strong Consistency
    
    ✅ **Use Cases**:
    - Financial transactions (account balances)
    - Inventory management (prevent overselling)
    - Coordination primitives (locks, leader election)
    - Regulatory compliance requiring audit trails
    
    ⚠️ **Cost**:
    - Higher latency (multiple network round-trips)
    - Reduced availability during partitions
    - More complex operations (read-modify-write)
    
    💡 **Example**: Google Spanner uses atomic clocks to achieve strong consistency globally
    
    ### When to Choose Eventual Consistency
    
    ✅ **Use Cases**:
    - Social media feeds (stale data acceptable)
    - Product catalogs (eventual propagation OK)
    - Analytics dashboards (approximate is fine)
    - Caching layers (staleness tolerable)
    
    ⚠️ **Cost**:
    - Application must handle conflicts
    - Users may see stale data
    - "Last write wins" may not match intent
    
    💡 **Example**: Amazon shopping cart uses eventual consistency for availability
    
    ### The Hybrid Approach
    
    Many modern systems use **mixed consistency**:
    
    ```python
    # Strong consistency for critical operations
    def book_room(room_id):
        with transaction(consistency='STRONG'):
            if room.is_available:
                room.book()  # Prevent double-booking
    
    # Eventual consistency for less critical
    def update_user_preferences(user_id, prefs):
        with transaction(consistency='EVENTUAL'):
            user.preferences = prefs  # OK if delayed
    ```
    
    This is called **RedBlue Consistency** or **Mixed Consistency** models.
    """)

with st.expander("🔬 Experiment Ideas"):
    st.markdown("""
    ### Try These Experiments
    
    1. **Latency vs Consistency**
       - Set N=5, toggle between Strong (R=3,W=3) and Weak (R=1,W=1)
       - Compare latencies: Strong is ~3x higher due to quorum coordination
    
    2. **CAP During Partitions**
       - Enable "Network Partition"
       - Strong Consistency: Watch operations fail (choosing C over A)
       - Eventual Consistency: Operations succeed but may return stale data (choosing A over C)
    
    3. **WAN vs LAN**
       - Network Latency = 5ms (data center): Quorum cost is moderate
       - Network Latency = 150ms (cross-region): Quorum cost is severe
       - Decision: Co-locate for strong consistency, or use eventual consistency for geo-distribution
    
    4. **Write-Heavy vs Read-Heavy**
       - Write Ratio = 0.8: Watch write latency impact
       - Write Ratio = 0.2: System performs better (reads can be served from any node)
    
    5. **Linearizable (R=W=N)**
       - Highest consistency but requires ALL nodes
       - Single node failure breaks the system
       - Compare with Strong (R=W=3): Can tolerate failures
    """)
