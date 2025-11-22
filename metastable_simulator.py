"""
Metastable Failure Simulator
Demonstrates how retry storms and slow error handling lead to metastable collapse
Based on AWS research: https://sigops.org/s/conferences/hotos/2025/papers/hotos25-106.pdf
"""

import streamlit as st
import simpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

class MetastableServer:
    """
    Server with latency that increases with concurrency (coordination overhead)
    """
    def __init__(self, env, base_latency, concurrency_slope, capacity, stats):
        self.env = env
        self.base_latency = base_latency
        self.concurrency_slope = concurrency_slope
        self.capacity = capacity
        self.stats = stats
        self.active_requests = 0
        self.semaphore = simpy.Resource(env, capacity=capacity)
        
    def current_latency(self):
        """Latency increases linearly with concurrency"""
        return self.base_latency + (self.active_requests * self.concurrency_slope)
    
    def process_request(self, request_id, arrival_time):
        """Process a single request"""
        with self.semaphore.request() as req:
            yield req
            self.active_requests += 1
            
            # Latency grows with concurrency
            processing_time = self.current_latency()
            yield self.env.timeout(processing_time)
            
            self.active_requests -= 1
            latency = self.env.now - arrival_time
            self.stats['latencies'].append(latency)
            self.stats['timestamps'].append(self.env.now)
            self.stats['concurrency'].append(self.active_requests)
            self.stats['successes'] += 1

class RetryClient:
    """Client with configurable retry behavior"""
    def __init__(self, env, server, timeout, max_retries, backoff_enabled, stats):
        self.env = env
        self.server = server
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_enabled = backoff_enabled
        self.stats = stats
        self.backoff_base = 0.1
        
    def make_request(self, request_id):
        """Attempt request with retries"""
        arrival_time = self.env.now
        retries = 0
        
        while retries <= self.max_retries:
            # Create timeout race condition
            request_proc = self.env.process(
                self.server.process_request(request_id, arrival_time)
            )
            timeout_proc = self.env.timeout(self.timeout)
            
            # Race between completion and timeout
            result = yield request_proc | timeout_proc
            
            if request_proc in result:
                # Success!
                return
            else:
                # Timeout occurred
                self.stats['timeouts'] += 1
                retries += 1
                
                if retries <= self.max_retries:
                    self.stats['retries'] += 1
                    # Exponential backoff with jitter if enabled
                    if self.backoff_enabled:
                        backoff = (self.backoff_base * (2 ** retries) * 
                                 random.uniform(0.5, 1.5))
                        yield self.env.timeout(backoff)
        
        # All retries exhausted
        self.stats['failures'] += 1

class LoadGenerator:
    """Generates load with configurable patterns"""
    def __init__(self, env, client, pattern='constant', base_rate=10, 
                 spike_rate=50, spike_start=20, spike_duration=10):
        self.env = env
        self.client = client
        self.pattern = pattern
        self.base_rate = base_rate
        self.spike_rate = spike_rate
        self.spike_start = spike_start
        self.spike_duration = spike_duration
        self.request_id = 0
        
    def get_current_rate(self):
        """Get arrival rate based on load pattern"""
        t = self.env.now
        
        if self.pattern == 'constant':
            return self.base_rate
        elif self.pattern == 'spike':
            if self.spike_start <= t < self.spike_start + self.spike_duration:
                return self.spike_rate
            return self.base_rate
        elif self.pattern == 'ramp':
            # Ramp up to peak and back down
            if t < 25:
                return self.base_rate + (self.spike_rate - self.base_rate) * (t / 25)
            elif t < 50:
                return self.spike_rate - (self.spike_rate - self.base_rate) * ((t - 25) / 25)
            return self.base_rate
        
    def generate(self):
        """Generate load according to pattern"""
        while True:
            rate = self.get_current_rate()
            if rate > 0:
                # Poisson arrival process
                yield self.env.timeout(random.expovariate(rate))
                self.request_id += 1
                self.env.process(self.client.make_request(self.request_id))

def run_simulation(duration, server_capacity, base_latency, concurrency_slope,
                   timeout, max_retries, backoff_enabled, load_pattern,
                   base_rate, spike_rate, spike_start, spike_duration):
    """Run the metastable failure simulation"""
    
    # Initialize
    env = simpy.Environment()
    stats = {
        'successes': 0,
        'timeouts': 0,
        'retries': 0,
        'failures': 0,
        'latencies': [],
        'timestamps': [],
        'concurrency': []
    }
    
    # Create components
    server = MetastableServer(env, base_latency, concurrency_slope, 
                              server_capacity, stats)
    client = RetryClient(env, server, timeout, max_retries, 
                        backoff_enabled, stats)
    generator = LoadGenerator(env, client, load_pattern, base_rate, 
                             spike_rate, spike_start, spike_duration)
    
    # Start load generation
    env.process(generator.generate())
    
    # Run simulation
    env.run(until=duration)
    
    return stats

def plot_results(stats, title):
    """Create comprehensive visualization of simulation results"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Server Concurrency Over Time', 
                       'Request Latency Distribution',
                       'Latency Over Time (P50, P95, P99)',
                       'System Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "indicator"}]]
    )
    
    # 1. Concurrency over time
    if stats['timestamps']:
        fig.add_trace(
            go.Scatter(x=stats['timestamps'], y=stats['concurrency'],
                      mode='lines', name='Active Requests',
                      line=dict(color='red')),
            row=1, col=1
        )
    
    # 2. Latency distribution
    if stats['latencies']:
        fig.add_trace(
            go.Histogram(x=stats['latencies'], nbinsx=50, name='Latency',
                        marker_color='blue', opacity=0.7),
            row=1, col=2
        )
    
    # 3. Latency percentiles over time
    if stats['timestamps'] and stats['latencies']:
        # Calculate rolling percentiles
        window = 50
        if len(stats['latencies']) >= window:
            p50s, p95s, p99s = [], [], []
            times = []
            for i in range(window, len(stats['latencies'])):
                window_data = stats['latencies'][i-window:i]
                p50s.append(np.percentile(window_data, 50))
                p95s.append(np.percentile(window_data, 95))
                p99s.append(np.percentile(window_data, 99))
                times.append(stats['timestamps'][i])
            
            fig.add_trace(
                go.Scatter(x=times, y=p50s, mode='lines', name='P50',
                          line=dict(color='green')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=times, y=p95s, mode='lines', name='P95',
                          line=dict(color='orange')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=times, y=p99s, mode='lines', name='P99',
                          line=dict(color='red')),
                row=2, col=1
            )
    
    # 4. Key metrics
    total_requests = (stats['successes'] + stats['failures'])
    success_rate = (stats['successes'] / total_requests * 100) if total_requests > 0 else 0
    avg_latency = np.mean(stats['latencies']) if stats['latencies'] else 0
    
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=success_rate,
            title={"text": f"Success Rate<br>{stats['successes']}/{total_requests}"},
            number={'suffix': "%"},
            domain={'x': [0, 0.5], 'y': [0, 0.5]}
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text=title,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Concurrent Requests", row=1, col=1)
    fig.update_xaxes(title_text="Latency (s)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Latency (s)", row=2, col=1)
    
    return fig

# Streamlit UI
st.set_page_config(layout="wide", page_title="Metastable Failure Simulator")

st.title("🔄 Metastable Failure Simulator")
st.markdown("""
This simulator demonstrates how **retry storms** and **coordination overhead** lead to 
**metastable failures** - where systems enter a self-sustaining degraded state.

**Key Concept**: As server concurrency increases, latency grows due to coordination costs.
This causes timeouts, which trigger retries, which increase concurrency further, creating
a feedback loop that can persist even after the initial stressor is removed.
""")

# Sidebar controls
st.sidebar.header("⚙️ Simulation Parameters")

st.sidebar.subheader("Server Configuration")
server_capacity = st.sidebar.slider("Server Capacity", 5, 100, 50, 
                                    help="Maximum concurrent requests")
base_latency = st.sidebar.slider("Base Latency (s)", 0.1, 2.0, 0.5, 0.1,
                                 help="Processing time with no contention")
concurrency_slope = st.sidebar.slider("Concurrency Slope", 0.001, 0.05, 0.01, 0.001,
                                     help="Additional latency per concurrent request")

st.sidebar.subheader("Client Retry Behavior")
timeout = st.sidebar.slider("Request Timeout (s)", 1.0, 10.0, 3.0, 0.5,
                            help="Time before client gives up on request")
max_retries = st.sidebar.slider("Max Retries", 0, 5, 3,
                                help="Number of retry attempts")
backoff_enabled = st.sidebar.checkbox("Enable Exponential Backoff", True,
                                      help="Add delay between retries")

st.sidebar.subheader("Load Pattern")
load_pattern = st.sidebar.selectbox("Pattern", ['constant', 'spike', 'ramp'])
base_rate = st.sidebar.slider("Base Rate (req/s)", 1, 50, 10)
spike_rate = st.sidebar.slider("Spike Rate (req/s)", base_rate, 100, 50)
spike_start = st.sidebar.slider("Spike Start (s)", 10, 40, 20)
spike_duration = st.sidebar.slider("Spike Duration (s)", 5, 30, 10)

duration = st.sidebar.slider("Simulation Duration (s)", 30, 120, 60)

# Run simulation button
if st.sidebar.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        stats = run_simulation(
            duration, server_capacity, base_latency, concurrency_slope,
            timeout, max_retries, backoff_enabled, load_pattern,
            base_rate, spike_rate, spike_start, spike_duration
        )
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Successes", stats['successes'])
        with col2:
            st.metric("⏱️ Timeouts", stats['timeouts'])
        with col3:
            st.metric("🔄 Retries", stats['retries'])
        with col4:
            st.metric("❌ Failures", stats['failures'])
        
        # Plot results
        fig = plot_results(stats, f"Metastable Failure Analysis - {load_pattern.title()} Load")
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis
        st.subheader("📊 Analysis")
        
        if stats['latencies']:
            avg_latency = np.mean(stats['latencies'])
            p99_latency = np.percentile(stats['latencies'], 99)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Latency", f"{avg_latency:.2f}s")
            with col2:
                st.metric("P99 Latency", f"{p99_latency:.2f}s")
            
            # Detect metastability
            if p99_latency > timeout * 0.8:
                st.error("""
                ⚠️ **Metastable Failure Detected!**
                
                P99 latency is approaching or exceeding the timeout threshold. This creates a 
                retry storm where:
                1. Requests timeout due to high latency
                2. Clients retry, adding more load
                3. Increased load causes even higher latency
                4. System enters self-sustaining degraded state
                
                **Mitigations:**
                - Enable exponential backoff with jitter
                - Implement admission control (reject requests when overloaded)
                - Add circuit breakers to stop retry storms
                - Reduce concurrency slope (better system architecture)
                """)
            elif stats['retries'] > stats['successes'] * 0.1:
                st.warning("""
                ⚠️ **High Retry Rate**
                
                Retry rate is high. While not fully metastable, the system is showing signs of stress.
                Consider enabling backoff or reducing load.
                """)
            else:
                st.success("""
                ✅ **System Stable**
                
                The system is handling load well. Retry mechanisms are working as intended.
                """)

# Educational content
with st.expander("📚 Learn More: What is Metastability?"):
    st.markdown("""
    ### Metastable Failures in Distributed Systems
    
    **Definition**: A metastable failure occurs when a system enters a degraded state due to 
    a temporary stressor, but fails to recover after the stressor is removed.
    
    **Why They Happen**:
    - **Sustaining Effects**: Retry storms, slow error handling, and coordination overhead
      create feedback loops
    - **Goodput Collapse**: System spends resources on non-productive work (retries, timeouts)
    - **Work Amplification**: Each failed request generates multiple retry attempts
    
    **Real-World Examples**:
    - AWS Kinesis outage (2020): Retry storms caused cascading failures
    - OpenAI ChatGPT outage: Database connection pool exhaustion led to sustained degradation
    
    **Key Insight from AWS Research**:
    > "Abstract models guide expensive tests, while concrete tests calibrate abstract models"
    
    The path to prevention: CTMC → Discrete Event Sim → Emulation → Stress Testing
    """)

with st.expander("🛡️ Prevention Strategies"):
    st.markdown("""
    ### How to Prevent Metastable Failures
    
    1. **Exponential Backoff with Jitter**
       - Add increasing delays between retries
       - Randomize delays to prevent thundering herd
    
    2. **Adaptive Retry Budgets**
       - Limit total retries across all clients
       - Use token bucket for retry capacity
    
    3. **Circuit Breakers**
       - Detect failure patterns and stop sending requests
       - Allow system to recover before resuming
    
    4. **Admission Control**
       - Reject new requests when queue is full
       - Better to fail fast than degrade for everyone
    
    5. **Load Shedding**
       - Drop lower-priority requests under stress
       - Preserve capacity for critical operations
    
    6. **System Architecture**
       - Reduce coordination overhead (lower concurrency slope)
       - Implement timeout hierarchies (upstream < downstream)
       - Design for graceful degradation
    """)
