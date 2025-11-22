"""
CRDT (Conflict-Free Replicated Data Types) Simulator
Demonstrates how CRDTs achieve strong eventual consistency without coordination
Shows different CRDT types: Counter, Set, Map, and their convergence properties
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Any
from enum import Enum
import time

class CRDTType(Enum):
    """Different types of CRDTs"""
    G_COUNTER = "G-Counter (Grow-Only)"
    PN_COUNTER = "PN-Counter (Increment/Decrement)"
    OR_SET = "OR-Set (Observed-Remove)"
    LWW_MAP = "LWW-Map (Last-Write-Wins)"

@dataclass
class Operation:
    """An operation on a CRDT"""
    replica_id: int
    timestamp: float
    op_type: str  # 'increment', 'decrement', 'add', 'remove', 'set'
    key: Any = None
    value: Any = None
    unique_id: str = field(default_factory=lambda: f"{time.time()}_{random.randint(0, 1000000)}")

class GCounter:
    """
    Grow-only Counter CRDT
    State: map from replica_id -> count
    Merge: take max of each replica's count
    """
    def __init__(self, replica_id: int):
        self.replica_id = replica_id
        self.counts: Dict[int, int] = {}
    
    def increment(self, amount: int = 1):
        """Increment this replica's counter"""
        current = self.counts.get(self.replica_id, 0)
        self.counts[self.replica_id] = current + amount
    
    def value(self) -> int:
        """Get total value across all replicas"""
        return sum(self.counts.values())
    
    def merge(self, other: 'GCounter'):
        """Merge with another counter (idempotent, commutative, associative)"""
        for replica_id, count in other.counts.items():
            self.counts[replica_id] = max(self.counts.get(replica_id, 0), count)
    
    def copy(self) -> 'GCounter':
        """Create a copy of this counter"""
        new_counter = GCounter(self.replica_id)
        new_counter.counts = self.counts.copy()
        return new_counter

class PNCounter:
    """
    Positive-Negative Counter CRDT
    State: two G-Counters (positive and negative)
    Value: positive - negative
    """
    def __init__(self, replica_id: int):
        self.replica_id = replica_id
        self.positive = GCounter(replica_id)
        self.negative = GCounter(replica_id)
    
    def increment(self, amount: int = 1):
        """Increment counter"""
        self.positive.increment(amount)
    
    def decrement(self, amount: int = 1):
        """Decrement counter"""
        self.negative.increment(amount)
    
    def value(self) -> int:
        """Get net value"""
        return self.positive.value() - self.negative.value()
    
    def merge(self, other: 'PNCounter'):
        """Merge with another PN-Counter"""
        self.positive.merge(other.positive)
        self.negative.merge(other.negative)
    
    def copy(self) -> 'PNCounter':
        """Create a copy"""
        new_counter = PNCounter(self.replica_id)
        new_counter.positive = self.positive.copy()
        new_counter.negative = self.negative.copy()
        return new_counter

class ORSet:
    """
    Observed-Remove Set CRDT
    Each element has unique tags (replica_id, timestamp)
    Add: add element with new tag
    Remove: remove all observed tags
    """
    def __init__(self, replica_id: int):
        self.replica_id = replica_id
        self.elements: Dict[Any, Set[str]] = {}  # element -> set of unique tags
    
    def add(self, element: Any, unique_tag: str):
        """Add element with unique tag"""
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(unique_tag)
    
    def remove(self, element: Any, observed_tags: Set[str]):
        """Remove element (only observed tags)"""
        if element in self.elements:
            self.elements[element] -= observed_tags
            if not self.elements[element]:
                del self.elements[element]
    
    def contains(self, element: Any) -> bool:
        """Check if element is in set"""
        return element in self.elements and len(self.elements[element]) > 0
    
    def get_elements(self) -> Set[Any]:
        """Get all elements in set"""
        return set(self.elements.keys())
    
    def get_tags(self, element: Any) -> Set[str]:
        """Get tags for an element"""
        return self.elements.get(element, set()).copy()
    
    def merge(self, other: 'ORSet'):
        """Merge with another OR-Set"""
        for element, tags in other.elements.items():
            if element not in self.elements:
                self.elements[element] = set()
            self.elements[element].update(tags)
    
    def copy(self) -> 'ORSet':
        """Create a copy"""
        new_set = ORSet(self.replica_id)
        for element, tags in self.elements.items():
            new_set.elements[element] = tags.copy()
        return new_set

class LWWMap:
    """
    Last-Write-Wins Map CRDT
    Each key has (value, timestamp, replica_id)
    Merge: take entry with highest timestamp
    """
    def __init__(self, replica_id: int):
        self.replica_id = replica_id
        self.entries: Dict[Any, Tuple[Any, float, int]] = {}  # key -> (value, timestamp, replica_id)
    
    def set(self, key: Any, value: Any, timestamp: float):
        """Set a key-value pair"""
        current = self.entries.get(key)
        if current is None or timestamp > current[1]:
            self.entries[key] = (value, timestamp, self.replica_id)
        elif timestamp == current[1] and self.replica_id > current[2]:
            # Tie-break by replica_id
            self.entries[key] = (value, timestamp, self.replica_id)
    
    def get(self, key: Any) -> Any:
        """Get value for key"""
        entry = self.entries.get(key)
        return entry[0] if entry else None
    
    def keys(self) -> Set[Any]:
        """Get all keys"""
        return set(self.entries.keys())
    
    def merge(self, other: 'LWWMap'):
        """Merge with another LWW-Map"""
        for key, (value, timestamp, replica_id) in other.entries.items():
            current = self.entries.get(key)
            if current is None:
                self.entries[key] = (value, timestamp, replica_id)
            else:
                curr_val, curr_ts, curr_replica = current
                if timestamp > curr_ts or (timestamp == curr_ts and replica_id > curr_replica):
                    self.entries[key] = (value, timestamp, replica_id)
    
    def copy(self) -> 'LWWMap':
        """Create a copy"""
        new_map = LWWMap(self.replica_id)
        new_map.entries = self.entries.copy()
        return new_map

class CRDTReplica:
    """A replica node with a CRDT"""
    def __init__(self, replica_id: int, crdt_type: CRDTType):
        self.replica_id = replica_id
        self.crdt_type = crdt_type
        
        # Initialize CRDT based on type
        if crdt_type == CRDTType.G_COUNTER:
            self.crdt = GCounter(replica_id)
        elif crdt_type == CRDTType.PN_COUNTER:
            self.crdt = PNCounter(replica_id)
        elif crdt_type == CRDTType.OR_SET:
            self.crdt = ORSet(replica_id)
        elif crdt_type == CRDTType.LWW_MAP:
            self.crdt = LWWMap(replica_id)
        
        # Track operations
        self.operations: List[Operation] = []
        self.merge_count = 0
    
    def apply_operation(self, op: Operation):
        """Apply an operation to this replica"""
        self.operations.append(op)
        
        if self.crdt_type == CRDTType.G_COUNTER:
            if op.op_type == 'increment':
                self.crdt.increment(op.value or 1)
        
        elif self.crdt_type == CRDTType.PN_COUNTER:
            if op.op_type == 'increment':
                self.crdt.increment(op.value or 1)
            elif op.op_type == 'decrement':
                self.crdt.decrement(op.value or 1)
        
        elif self.crdt_type == CRDTType.OR_SET:
            if op.op_type == 'add':
                self.crdt.add(op.value, op.unique_id)
            elif op.op_type == 'remove':
                # Get current tags and remove
                tags = self.crdt.get_tags(op.value)
                self.crdt.remove(op.value, tags)
        
        elif self.crdt_type == CRDTType.LWW_MAP:
            if op.op_type == 'set':
                self.crdt.set(op.key, op.value, op.timestamp)
    
    def merge_with(self, other: 'CRDTReplica'):
        """Merge with another replica"""
        self.crdt.merge(other.crdt)
        self.merge_count += 1
    
    def get_state(self):
        """Get current state of CRDT"""
        if self.crdt_type == CRDTType.G_COUNTER:
            return self.crdt.value()
        elif self.crdt_type == CRDTType.PN_COUNTER:
            return self.crdt.value()
        elif self.crdt_type == CRDTType.OR_SET:
            return self.crdt.get_elements()
        elif self.crdt_type == CRDTType.LWW_MAP:
            return {k: self.crdt.get(k) for k in self.crdt.keys()}
    
    def copy(self) -> 'CRDTReplica':
        """Create a copy of this replica"""
        new_replica = CRDTReplica(self.replica_id, self.crdt_type)
        new_replica.crdt = self.crdt.copy()
        new_replica.operations = self.operations.copy()
        new_replica.merge_count = self.merge_count
        return new_replica

def simulate_crdt_system(crdt_type: CRDTType, n_replicas: int, n_operations: int,
                         merge_probability: float, network_delay: float):
    """Simulate a distributed CRDT system"""
    
    # Create replicas
    replicas = [CRDTReplica(i, crdt_type) for i in range(n_replicas)]
    
    # Track history for visualization
    history = []
    current_time = 0.0
    
    # Generate random operations
    for i in range(n_operations):
        current_time += random.uniform(0.1, 1.0)
        
        # Pick a random replica to perform operation
        replica_idx = random.randint(0, n_replicas - 1)
        replica = replicas[replica_idx]
        
        # Generate operation based on CRDT type
        if crdt_type == CRDTType.G_COUNTER:
            op = Operation(
                replica_id=replica_idx,
                timestamp=current_time,
                op_type='increment',
                value=random.randint(1, 5)
            )
        
        elif crdt_type == CRDTType.PN_COUNTER:
            op_type = random.choice(['increment', 'decrement'])
            op = Operation(
                replica_id=replica_idx,
                timestamp=current_time,
                op_type=op_type,
                value=random.randint(1, 5)
            )
        
        elif crdt_type == CRDTType.OR_SET:
            elements = ['apple', 'banana', 'cherry', 'date', 'elderberry']
            op_type = random.choice(['add', 'add', 'remove'])  # Bias toward adds
            element = random.choice(elements)
            op = Operation(
                replica_id=replica_idx,
                timestamp=current_time,
                op_type=op_type,
                value=element
            )
        
        elif crdt_type == CRDTType.LWW_MAP:
            keys = ['user1', 'user2', 'user3', 'user4']
            key = random.choice(keys)
            value = f"value_{random.randint(1, 100)}"
            op = Operation(
                replica_id=replica_idx,
                timestamp=current_time,
                op_type='set',
                key=key,
                value=value
            )
        
        # Apply operation
        replica.apply_operation(op)
        
        # Record state of all replicas
        states = [r.get_state() for r in replicas]
        
        # Check convergence (all replicas have same state)
        if crdt_type in [CRDTType.G_COUNTER, CRDTType.PN_COUNTER]:
            converged = len(set(states)) == 1
        elif crdt_type == CRDTType.OR_SET:
            converged = len(set(frozenset(s) for s in states)) == 1
        elif crdt_type == CRDTType.LWW_MAP:
            converged = len(set(frozenset(s.items()) for s in states)) == 1
        
        history.append({
            'time': current_time,
            'operation': f"{op.op_type} by R{replica_idx}",
            'states': [str(s) for s in states],
            'converged': converged,
            'replica_id': replica_idx
        })
        
        # Probabilistically trigger merges (gossip protocol)
        if random.random() < merge_probability:
            # Random pairs of replicas merge
            for _ in range(random.randint(1, n_replicas // 2)):
                idx1, idx2 = random.sample(range(n_replicas), 2)
                
                # Simulate network delay
                current_time += network_delay
                
                # Bidirectional merge
                replicas[idx1].merge_with(replicas[idx2])
                replicas[idx2].merge_with(replicas[idx1])
            
            # Record state after merge
            states = [r.get_state() for r in replicas]
            
            if crdt_type in [CRDTType.G_COUNTER, CRDTType.PN_COUNTER]:
                converged = len(set(states)) == 1
            elif crdt_type == CRDTType.OR_SET:
                converged = len(set(frozenset(s) for s in states)) == 1
            elif crdt_type == CRDTType.LWW_MAP:
                converged = len(set(frozenset(s.items()) for s in states)) == 1
            
            history.append({
                'time': current_time,
                'operation': 'MERGE',
                'states': [str(s) for s in states],
                'converged': converged,
                'replica_id': -1  # Merge event
            })
    
    return history, replicas

def plot_crdt_convergence(history: List[Dict], n_replicas: int, crdt_type: CRDTType):
    """Visualize CRDT convergence over time"""
    
    df = pd.DataFrame(history)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Replica States Over Time', 'Convergence Status'),
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # 1. State evolution for each replica
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    
    for replica_id in range(n_replicas):
        # Extract state for this replica
        y_values = []
        x_values = []
        
        for idx, row in df.iterrows():
            x_values.append(row['time'])
            
            # Parse state based on CRDT type
            state_str = row['states'][replica_id]
            
            if crdt_type in [CRDTType.G_COUNTER, CRDTType.PN_COUNTER]:
                # Counter: extract numeric value
                y_values.append(int(state_str))
            elif crdt_type == CRDTType.OR_SET:
                # Set: use size
                y_values.append(len(eval(state_str)))
            elif crdt_type == CRDTType.LWW_MAP:
                # Map: use size
                y_values.append(len(eval(state_str)))
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=f'Replica {replica_id}',
                line=dict(color=colors[replica_id % len(colors)]),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
    
    # Mark merge events
    merge_times = [row['time'] for _, row in df.iterrows() if row['operation'] == 'MERGE']
    if merge_times:
        for merge_time in merge_times:
            fig.add_vline(
                x=merge_time,
                line_dash="dash",
                line_color="gray",
                opacity=0.3,
                row=1, col=1
            )
    
    # 2. Convergence indicator
    convergence_values = [1 if row['converged'] else 0 for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=convergence_values,
            mode='lines',
            fill='tozeroy',
            name='Converged',
            line=dict(color='green'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        title_text=f"CRDT Convergence: {crdt_type.value}",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    if crdt_type in [CRDTType.G_COUNTER, CRDTType.PN_COUNTER]:
        fig.update_yaxes(title_text="Counter Value", row=1, col=1)
    elif crdt_type == CRDTType.OR_SET:
        fig.update_yaxes(title_text="Set Size", row=1, col=1)
    elif crdt_type == CRDTType.LWW_MAP:
        fig.update_yaxes(title_text="Map Size", row=1, col=1)
    
    fig.update_yaxes(title_text="Converged", row=2, col=1)
    
    return fig

# Streamlit UI
st.set_page_config(layout="wide", page_title="CRDT Simulator")

st.title("🔀 CRDT (Conflict-Free Replicated Data Types) Simulator")

st.markdown("""
Explore how **CRDTs** achieve **strong eventual consistency** without coordination!

**Key Principles**:
- **No coordination required**: Updates applied independently
- **Mathematical convergence guarantee**: All replicas eventually reach the same state
- **Commutative operations**: Order doesn't matter
- **Idempotent merges**: Can merge multiple times safely

**Trade-off**: Eventual consistency (not linearizable) for high availability and low latency
""")

# Controls
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔧 CRDT Configuration")
    
    crdt_type = st.selectbox(
        "CRDT Type",
        [CRDTType.G_COUNTER, CRDTType.PN_COUNTER, CRDTType.OR_SET, CRDTType.LWW_MAP],
        format_func=lambda x: x.value
    )
    
    n_replicas = st.slider("Number of Replicas", 2, 8, 3)
    n_operations = st.slider("Number of Operations", 10, 100, 30)

with col2:
    st.subheader("🌐 Network Configuration")
    
    merge_probability = st.slider(
        "Merge Probability",
        0.0, 1.0, 0.3, 0.1,
        help="Probability of gossip/merge after each operation"
    )
    
    network_delay = st.slider(
        "Network Delay (s)",
        0.0, 2.0, 0.5, 0.1,
        help="Time for replicas to exchange state"
    )

# CRDT type descriptions
crdt_descriptions = {
    CRDTType.G_COUNTER: """
    **G-Counter (Grow-Only Counter)**
    - Can only increment
    - Each replica tracks its own increments
    - Merge: take max of each replica's count
    - Use case: Page views, likes (monotonic)
    """,
    
    CRDTType.PN_COUNTER: """
    **PN-Counter (Positive-Negative Counter)**
    - Can increment and decrement
    - Two G-Counters: positive and negative
    - Value = positive - negative
    - Use case: Inventory, bank balance (eventually)
    """,
    
    CRDTType.OR_SET: """
    **OR-Set (Observed-Remove Set)**
    - Add elements with unique tags
    - Remove only observed tags
    - Concurrent add beats remove
    - Use case: Shopping cart, collaborative lists
    """,
    
    CRDTType.LWW_MAP: """
    **LWW-Map (Last-Write-Wins Map)**
    - Each entry has timestamp
    - Merge: keep entry with latest timestamp
    - Tie-break by replica ID
    - Use case: User profiles, configuration
    """
}

st.info(crdt_descriptions[crdt_type])

# Run simulation
if st.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Simulating CRDT system..."):
        history, replicas = simulate_crdt_system(
            crdt_type, n_replicas, n_operations,
            merge_probability, network_delay
        )
        
        # Metrics
        final_states = [r.get_state() for r in replicas]
        
        # Check final convergence
        if crdt_type in [CRDTType.G_COUNTER, CRDTType.PN_COUNTER]:
            converged = len(set(final_states)) == 1
            final_value = final_states[0] if converged else "DIVERGED"
        elif crdt_type == CRDTType.OR_SET:
            converged = len(set(frozenset(s) for s in final_states)) == 1
            final_value = final_states[0] if converged else "DIVERGED"
        elif crdt_type == CRDTType.LWW_MAP:
            converged = len(set(frozenset(s.items()) for s in final_states)) == 1
            final_value = final_states[0] if converged else "DIVERGED"
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Operations", n_operations)
        with col2:
            total_merges = sum(r.merge_count for r in replicas)
            st.metric("Total Merges", total_merges)
        with col3:
            convergence_rate = sum(1 for h in history if h['converged']) / len(history) * 100
            st.metric("Convergence Rate", f"{convergence_rate:.1f}%")
        with col4:
            if converged:
                st.metric("Final State", "✅ CONVERGED", delta="All replicas agree")
            else:
                st.metric("Final State", "⚠️ DIVERGED", delta="Need more merges")
        
        # Plot convergence
        fig = plot_crdt_convergence(history, n_replicas, crdt_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show final states
        st.subheader("📊 Final States")
        
        state_df = pd.DataFrame([
            {
                'Replica': f"Replica {i}",
                'Final State': str(state),
                'Operations Applied': len(replicas[i].operations),
                'Merges Performed': replicas[i].merge_count
            }
            for i, state in enumerate(final_states)
        ])
        
        st.dataframe(state_df, use_container_width=True)
        
        # Analysis
        st.subheader("🔍 Analysis")
        
        if converged:
            st.success(f"""
            ✅ **Perfect Convergence Achieved!**
            
            All {n_replicas} replicas have converged to the same state despite:
            - Operations applied to different replicas
            - Operations arriving in different orders
            - Network delays between merges
            
            **Final converged state**: `{final_value}`
            
            This demonstrates the **Strong Eventual Consistency** guarantee of CRDTs:
            - All replicas that have received the same set of updates will have the same state
            - No coordination or consensus protocol required
            - Mathematically guaranteed convergence
            """)
        else:
            st.warning(f"""
            ⚠️ **Convergence Not Yet Achieved**
            
            Replicas have diverged because they haven't exchanged all updates yet.
            
            **What's happening**:
            - Different replicas have different subsets of operations
            - More merges (gossip) are needed to propagate all updates
            
            **Solutions**:
            - Increase merge probability (gossip frequency)
            - Wait longer for updates to propagate
            - In real systems: background gossip ensures eventual convergence
            
            **Current replica states**:
            {chr(10).join(f"- Replica {i}: {state}" for i, state in enumerate(final_states))}
            """)
        
        # Theoretical analysis
        with st.expander("📐 Mathematical Properties"):
            st.markdown(f"""
            ### Why CRDTs Work: Mathematical Guarantees
            
            For **{crdt_type.value}**:
            
            **Commutativity**: Order of operations doesn't matter
            ```
            merge(A, merge(B, C)) = merge(merge(A, B), C)  ✓
            merge(A, B) = merge(B, A)  ✓
            ```
            
            **Idempotence**: Can merge multiple times safely
            ```
            merge(A, A) = A  ✓
            ```
            
            **Associativity**: Grouping doesn't matter
            ```
            merge(A, merge(B, C)) = merge(merge(A, B), C)  ✓
            ```
            
            These properties form a **join-semilattice**, which guarantees convergence!
            
            ### Strong Eventual Consistency (SEC)
            
            Three requirements (all met by CRDTs):
            1. **Eventual Delivery**: All updates eventually reach all replicas
            2. **Convergence**: Replicas with same updates have same state
            3. **Termination**: Operations complete in finite time
            
            ### Comparison with Quorum Systems
            
            | Property | CRDTs | Quorum (R+W>N) |
            |----------|-------|----------------|
            | Consistency | Eventual | Strong/Linearizable |
            | Coordination | None | Required |
            | Latency | Low (local) | High (network round-trips) |
            | Availability | High | Medium (needs quorum) |
            | Use Cases | High-write, AP | Strong guarantees, CP |
            """)

# Educational sections
with st.expander("📚 CRDT Deep Dive"):
    st.markdown("""
    ### Types of CRDTs
    
    **State-based CRDTs (CvRDTs)**:
    - Send full state to other replicas
    - Merge function must be commutative, associative, idempotent
    - Simple but potentially high bandwidth
    
    **Operation-based CRDTs (CmRDTs)**:
    - Send only operations to other replicas
    - Requires reliable, causally-ordered delivery
    - More efficient but more complex
    
    **Delta-state CRDTs (δ-CRDTs)**:
    - Best of both: send only the "delta" (changes)
    - Most practical for production systems
    
    ### Real-World CRDT Implementations
    
    **Databases**:
    - Redis Enterprise (CRDBs): Multi-master with CRDTs
    - Riak: Distributed key-value with CRDT data types
    - Azure Cosmos DB: Conflict resolution with LWW
    
    **Collaborative Editing**:
    - Automerge: CRDT library for JSON documents
    - Yjs: CRDT framework for real-time collaboration
    - Google Docs: Uses operational transforms (similar concept)
    
    **Distributed Systems**:
    - SoundCloud: Audio distribution platform
    - League of Legends: Game state synchronization
    - TomTom: Live traffic data aggregation
    
    ### Limitations and Challenges
    
    ⚠️ **Not all problems fit CRDTs**:
    - Cannot prevent double-booking (need coordination)
    - Cannot enforce constraints (e.g., unique usernames)
    - Semantic conflicts (e.g., two people claim same resource)
    
    ⚠️ **Design challenges**:
    - Merge semantics may not match user expectations
    - Garbage collection (tombstones grow forever)
    - Causal ordering required for some CRDTs
    
    ⚠️ **Security concerns**:
    - Byzantine fault tolerance (malicious replicas)
    - Access control more complex
    - Requires all devices to be trusted
    
    ### When to Use CRDTs
    
    ✅ **Good for**:
    - Shopping carts (items can be added concurrently)
    - Collaborative documents (multiple editors)
    - Distributed counters (analytics)
    - Eventually consistent databases (high write load)
    - Offline-first applications (sync when online)
    
    ❌ **Not good for**:
    - Bank account balance (need atomic transactions)
    - Inventory management (prevent overselling)
    - Access control (need immediate consistency)
    - Anything requiring coordination or constraints
    """)

with st.expander("🎯 Try These Experiments"):
    st.markdown("""
    ### Experiment 1: Convergence Speed
    - Set merge probability to 0.1 (infrequent gossip)
    - Watch replicas diverge
    - Increase to 0.8: replicas converge quickly
    - **Lesson**: Gossip frequency trades latency for consistency window
    
    ### Experiment 2: Network Partition
    - Use 4 replicas with low merge probability
    - Imagine replicas 0,1 can't reach replicas 2,3
    - After partition heals (merges resume), all converge
    - **Lesson**: CRDTs handle partitions gracefully (AP in CAP)
    
    ### Experiment 3: Concurrent Conflicts
    - Use OR-Set with 3 replicas
    - Watch how concurrent add/remove resolves
    - OR-Set semantics: add wins over remove
    - **Lesson**: CRDT semantics determine conflict resolution
    
    ### Experiment 4: LWW Limitations
    - Use LWW-Map with 2 replicas
    - Both set same key concurrently with different values
    - **Problem**: One value "wins" arbitrarily based on timestamp
    - **Lesson**: LWW can lose data! Not suitable for all use cases
    
    ### Experiment 5: Counter Comparison
    - Compare G-Counter (only increment) vs PN-Counter
    - G-Counter: Simpler, always grows
    - PN-Counter: More flexible, but more complex state
    - **Lesson**: Choose simplest CRDT that meets requirements
    """)

with st.expander("💡 Design Patterns"):
    st.markdown("""
    ### Pattern 1: Hybrid Consistency
    
    Use strong consistency for critical paths, CRDTs for everything else:
    
    ```python
    # Critical: Prevent double-booking (use transaction)
    def reserve_room(room_id):
        with transaction(consistency='STRONG'):
            if room.available:
                room.book()
    
    # Non-critical: User preferences (use CRDT)
    def update_preferences(user_id, prefs):
        user_prefs[user_id].merge(prefs)  # LWW-Map CRDT
    ```
    
    ### Pattern 2: CRDT + Consensus
    
    Use CRDTs for local updates, consensus for global invariants:
    
    ```python
    # Fast local updates with CRDT
    shopping_cart.add(item)  # OR-Set: Instant
    
    # Consensus for checkout (prevent overselling)
    def checkout():
        with consensus():
            for item in cart:
                if not inventory.reserve(item):
                    raise OutOfStockError
    ```
    
    ### Pattern 3: Causal Consistency
    
    Maintain causal order with vector clocks:
    
    ```python
    # Vector clock tracks causality
    def publish_post(post, vector_clock):
        # Post with causal metadata
        crdt.add(post, vc=vector_clock)
    
    # Display respects causality
    def get_timeline():
        return crdt.get_causally_ordered()
    ```
    
    ### Pattern 4: Garbage Collection
    
    CRDTs can grow forever without GC:
    
    ```python
    # Problem: OR-Set keeps all tombstones
    set.add("item")   # Add tag
    set.remove("item")  # Keep tombstone
    # Memory grows!
    
    # Solution: Periodic GC with version vectors
    if all_replicas_beyond_version(v):
        gc_tombstones_before(v)
    ```
    """)
