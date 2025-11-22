"""
Cross-System Interaction (CSI) Failure Simulator
Demonstrates how independently working systems fail through interaction discrepancies
Based on research: http://dprg.cs.uiuc.edu/data/files/2023/eurosys23-fall-final-CSI.pdf
"""

import streamlit as st
import simpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum

class FailureType(Enum):
    """Types of CSI failures"""
    METADATA_MISMATCH = "Metadata Interpretation Mismatch"
    SCHEMA_CONFLICT = "Data Schema Conflict"
    CONFIG_INCOHERENCE = "Configuration Incoherence"
    API_SEMANTIC_VIOLATION = "API Semantic Violation"
    TYPE_CONFUSION = "Type Confusion"

@dataclass
class DataSchema:
    """Schema for data exchange between systems"""
    schema_id: str
    fields: Dict[str, str]  # field_name -> type
    encoding: str  # 'utf-8', 'latin-1', etc.
    nullable_fields: set
    
    def is_compatible(self, other: 'DataSchema') -> bool:
        """Check schema compatibility"""
        # Simplified compatibility check
        return (self.fields == other.fields and 
                self.encoding == other.encoding)

@dataclass
class DataRecord:
    """A data record exchanged between systems"""
    data: Dict[str, Any]
    schema: DataSchema
    metadata: Dict[str, Any]
    
    def serialize(self, target_schema: DataSchema) -> Optional['DataRecord']:
        """Serialize to target schema"""
        try:
            # Check type compatibility
            serialized_data = {}
            for field, value in self.data.items():
                if field not in target_schema.fields:
                    # Field missing in target schema
                    return None
                
                expected_type = target_schema.fields[field]
                actual_type = self.schema.fields[field]
                
                if expected_type != actual_type:
                    # Type mismatch - simulate type coercion failures
                    if random.random() < 0.3:  # 30% fail on type mismatch
                        return None
                
                serialized_data[field] = value
            
            # Check encoding compatibility
            if self.schema.encoding != target_schema.encoding:
                # Encoding mismatch can cause data corruption
                if random.random() < 0.2:  # 20% fail on encoding mismatch
                    return None
            
            return DataRecord(serialized_data, target_schema, self.metadata.copy())
        except Exception:
            return None

class System:
    """A system in the distributed architecture"""
    def __init__(self, env, name: str, schema: DataSchema, 
                 processing_time: float, failure_rate: float = 0.0):
        self.env = env
        self.name = name
        self.schema = schema
        self.processing_time = processing_time
        self.failure_rate = failure_rate
        
        # Configuration
        self.config = {}
        
        # Stats
        self.requests_processed = 0
        self.requests_failed = 0
        self.csi_failures = 0
        
    def process_request(self, record: DataRecord, expected_config: Dict = None):
        """Process a request from another system"""
        yield self.env.timeout(self.processing_time)
        
        self.requests_processed += 1
        
        # Check for random failures
        if random.random() < self.failure_rate:
            self.requests_failed += 1
            return None
        
        # Check schema compatibility (data plane CSI)
        if not self.schema.is_compatible(record.schema):
            # Attempt to serialize/deserialize
            converted = record.serialize(self.schema)
            if converted is None:
                self.csi_failures += 1
                self.requests_failed += 1
                return None
            record = converted
        
        # Check configuration coherence (management plane CSI)
        if expected_config:
            for key, expected_value in expected_config.items():
                actual_value = self.config.get(key)
                if actual_value != expected_value:
                    # Configuration mismatch!
                    if random.random() < 0.4:  # 40% fail on config mismatch
                        self.csi_failures += 1
                        self.requests_failed += 1
                        return None
        
        # Process successfully
        return record
    
    def set_config(self, key: str, value: Any):
        """Set a configuration value"""
        self.config[key] = value

class DataPipelineSimulator:
    """
    Simulates a data pipeline with multiple systems
    demonstrating CSI failures
    """
    def __init__(self, env, failure_scenario: FailureType):
        self.env = env
        self.failure_scenario = failure_scenario
        self.stats = {
            'successful_requests': 0,
            'failed_requests': 0,
            'csi_failures': 0,
            'timestamps': [],
            'failure_points': [],
            'system_failures': {}
        }
        
        # Create systems based on failure scenario
        self._setup_scenario()
    
    def _setup_scenario(self):
        """Setup systems for different failure scenarios"""
        
        if self.failure_scenario == FailureType.METADATA_MISMATCH:
            # Scenario: Hive and Spark with different schema interpretations
            hive_schema = DataSchema(
                schema_id="hive_v1",
                fields={'id': 'int', 'name': 'string', 'timestamp': 'long'},
                encoding='utf-8',
                nullable_fields={'name'}
            )
            
            # Spark expects slightly different schema
            spark_schema = DataSchema(
                schema_id="spark_v1",
                fields={'id': 'int', 'name': 'string', 'timestamp': 'timestamp'},  # Different type!
                encoding='utf-8',
                nullable_fields={'name'}
            )
            
            self.system_a = System(self.env, "Hive", hive_schema, 0.1)
            self.system_b = System(self.env, "Spark", spark_schema, 0.1)
            
        elif self.failure_scenario == FailureType.SCHEMA_CONFLICT:
            # Scenario: Producer and Consumer with encoding mismatch
            producer_schema = DataSchema(
                schema_id="producer_v1",
                fields={'id': 'int', 'data': 'string'},
                encoding='latin-1',  # Different encoding!
                nullable_fields=set()
            )
            
            consumer_schema = DataSchema(
                schema_id="consumer_v1",
                fields={'id': 'int', 'data': 'string'},
                encoding='utf-8',  # Expected UTF-8
                nullable_fields=set()
            )
            
            self.system_a = System(self.env, "Producer", producer_schema, 0.05)
            self.system_b = System(self.env, "Consumer", consumer_schema, 0.05)
            
        elif self.failure_scenario == FailureType.CONFIG_INCOHERENCE:
            # Scenario: Systems with inconsistent configuration
            schema = DataSchema(
                schema_id="common_v1",
                fields={'id': 'int', 'value': 'string'},
                encoding='utf-8',
                nullable_fields=set()
            )
            
            self.system_a = System(self.env, "ServiceA", schema, 0.1)
            self.system_b = System(self.env, "ServiceB", schema, 0.1)
            
            # Set conflicting configs
            self.system_a.set_config("max_retries", 3)
            self.system_a.set_config("timeout", 30)
            
            # System B expects different config values
            self.system_b.set_config("max_retries", 5)  # Mismatch!
            self.system_b.set_config("timeout", 60)      # Mismatch!
            
        elif self.failure_scenario == FailureType.API_SEMANTIC_VIOLATION:
            # Scenario: API contract violation
            schema = DataSchema(
                schema_id="api_v1",
                fields={'request_id': 'string', 'data': 'string'},
                encoding='utf-8',
                nullable_fields=set()
            )
            
            # System A calls System B with assumptions about synchrony
            self.system_a = System(self.env, "Upstream", schema, 0.05)
            self.system_b = System(self.env, "Downstream", schema, 0.2, failure_rate=0.15)
            
        else:  # TYPE_CONFUSION
            # Scenario: Type interpretation mismatch
            schema_a = DataSchema(
                schema_id="v1",
                fields={'id': 'int', 'amount': 'float'},  # Float in system A
                encoding='utf-8',
                nullable_fields=set()
            )
            
            schema_b = DataSchema(
                schema_id="v1",
                fields={'id': 'int', 'amount': 'decimal'},  # Decimal in system B
                encoding='utf-8',
                nullable_fields=set()
            )
            
            self.system_a = System(self.env, "SystemA", schema_a, 0.1)
            self.system_b = System(self.env, "SystemB", schema_b, 0.1)
    
    def process_request(self, request_id: int):
        """Process a request through the pipeline"""
        start_time = self.env.now
        
        # Create initial record
        if self.failure_scenario == FailureType.METADATA_MISMATCH:
            data = {'id': request_id, 'name': f'user_{request_id}', 'timestamp': int(start_time * 1000)}
        elif self.failure_scenario == FailureType.SCHEMA_CONFLICT:
            data = {'id': request_id, 'data': f'data_{request_id}'}
        elif self.failure_scenario == FailureType.CONFIG_INCOHERENCE:
            data = {'id': request_id, 'value': f'value_{request_id}'}
        elif self.failure_scenario == FailureType.API_SEMANTIC_VIOLATION:
            data = {'request_id': str(request_id), 'data': f'payload_{request_id}'}
        else:  # TYPE_CONFUSION
            data = {'id': request_id, 'amount': 123.45}
        
        record = DataRecord(data, self.system_a.schema, {'request_id': request_id})
        
        # Process in System A
        result = yield self.env.process(self.system_a.process_request(record))
        
        if result is None:
            self.stats['failed_requests'] += 1
            self.stats['timestamps'].append(self.env.now)
            self.stats['failure_points'].append(self.system_a.name)
            self._record_system_failure(self.system_a.name)
            return
        
        # Pass to System B (with potential config expectations)
        expected_config = None
        if self.failure_scenario == FailureType.CONFIG_INCOHERENCE:
            # System B expects System A's config
            expected_config = {'max_retries': 3, 'timeout': 30}
        
        result = yield self.env.process(
            self.system_b.process_request(result, expected_config)
        )
        
        if result is None:
            self.stats['failed_requests'] += 1
            self.stats['timestamps'].append(self.env.now)
            self.stats['failure_points'].append(self.system_b.name)
            self._record_system_failure(self.system_b.name)
            return
        
        # Success!
        self.stats['successful_requests'] += 1
        self.stats['timestamps'].append(self.env.now)
    
    def _record_system_failure(self, system_name: str):
        """Record failure for a specific system"""
        if system_name not in self.stats['system_failures']:
            self.stats['system_failures'][system_name] = 0
        self.stats['system_failures'][system_name] += 1
        
        # Check if it's a CSI failure
        if system_name == self.system_b.name:
            if self.system_b.csi_failures > 0:
                self.stats['csi_failures'] += 1

class LoadGenerator:
    """Generate requests for the pipeline"""
    def __init__(self, env, pipeline: DataPipelineSimulator, rate: float):
        self.env = env
        self.pipeline = pipeline
        self.rate = rate
        self.request_id = 0
    
    def generate(self):
        """Generate load"""
        while True:
            yield self.env.timeout(random.expovariate(self.rate))
            self.request_id += 1
            self.env.process(self.pipeline.process_request(self.request_id))

def run_csi_simulation(failure_scenario: FailureType, duration: float, request_rate: float):
    """Run the CSI failure simulation"""
    env = simpy.Environment()
    
    pipeline = DataPipelineSimulator(env, failure_scenario)
    generator = LoadGenerator(env, pipeline, request_rate)
    
    env.process(generator.generate())
    env.run(until=duration)
    
    # Aggregate stats
    total_requests = (pipeline.stats['successful_requests'] + 
                     pipeline.stats['failed_requests'])
    
    pipeline.stats['total_requests'] = total_requests
    pipeline.stats['success_rate'] = (
        (pipeline.stats['successful_requests'] / total_requests * 100)
        if total_requests > 0 else 0
    )
    
    return pipeline.stats, pipeline

def plot_csi_results(stats: Dict, scenario: FailureType):
    """Visualize CSI failure results"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Success vs Failure Over Time',
            'Failure Distribution by System',
            'CSI Failure Impact',
            'Overall Metrics'
        ),
        specs=[
            [{"secondary_y": False}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "indicator"}]
        ]
    )
    
    # 1. Success/Failure over time
    if stats['timestamps']:
        # Create cumulative counts
        successes = []
        failures = []
        times = sorted(stats['timestamps'])
        
        success_count = 0
        failure_count = 0
        
        for t in times:
            if t in stats['timestamps'][:stats['successful_requests']]:
                success_count += 1
            else:
                failure_count += 1
            successes.append(success_count)
            failures.append(failure_count)
        
        fig.add_trace(
            go.Scatter(x=times, y=successes, name='Successful',
                      mode='lines', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=times, y=failures, name='Failed',
                      mode='lines', line=dict(color='red')),
            row=1, col=1
        )
    
    # 2. Failure by system
    if stats['system_failures']:
        systems = list(stats['system_failures'].keys())
        failures = list(stats['system_failures'].values())
        
        fig.add_trace(
            go.Bar(x=systems, y=failures, marker_color=['red', 'orange'],
                  text=failures, textposition='auto'),
            row=1, col=2
        )
    
    # 3. CSI vs Other failures
    csi_failures = stats.get('csi_failures', 0)
    other_failures = stats['failed_requests'] - csi_failures
    
    fig.add_trace(
        go.Bar(
            x=['CSI Failures', 'Other Failures'],
            y=[csi_failures, other_failures],
            marker_color=['darkred', 'orange'],
            text=[csi_failures, other_failures],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 4. Success rate indicator
    success_rate = stats.get('success_rate', 0)
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=success_rate,
            title={'text': "Success Rate"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if success_rate > 80 else "orange" if success_rate > 50 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        title_text=f"CSI Failure Analysis: {scenario.value}",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Count", row=1, col=1)
    fig.update_xaxes(title_text="System", row=1, col=2)
    fig.update_yaxes(title_text="Failures", row=1, col=2)
    fig.update_xaxes(title_text="Failure Type", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    return fig

# Streamlit UI
st.set_page_config(layout="wide", page_title="CSI Failure Simulator")

st.title("🔗 Cross-System Interaction (CSI) Failure Simulator")

st.markdown("""
Explore how **independently correct systems fail through their interactions**.

**Key Insight from Research**: 20% of catastrophic cloud incidents and 37% of open-source 
distributed system failures are CSI failures - where neither system is buggy on its own, 
but discrepancies in how they interact cause failures.

**Why CSI Failures Matter**:
- Cannot be detected by testing systems in isolation
- Emerge from subtle incompatibilities (schema, config, API semantics)
- 74% manifest as crashes (existing fault tolerance doesn't help)
- Single points of failure despite component redundancy
""")

# Failure scenario selection
st.subheader("🎯 Select Failure Scenario")

scenario_descriptions = {
    FailureType.METADATA_MISMATCH: {
        'title': '🗂️ Metadata Interpretation Mismatch',
        'description': """
        **Example**: Hive stores timestamp as `long`, Spark expects `timestamp` type.
        
        **Real Impact**: 82% of data-plane CSI failures in study.
        
        **Root Cause**: No unified schema enforcement across systems.
        """,
        'systems': 'Hive → Spark'
    },
    
    FailureType.SCHEMA_CONFLICT: {
        'title': '🔤 Data Schema & Encoding Conflict',
        'description': """
        **Example**: Producer uses Latin-1 encoding, Consumer expects UTF-8.
        
        **Real Impact**: Silent data corruption, difficult to debug.
        
        **Root Cause**: Ad-hoc serialization, inconsistent conventions.
        """,
        'systems': 'Producer → Consumer'
    },
    
    FailureType.CONFIG_INCOHERENCE: {
        'title': '⚙️ Configuration Incoherence',
        'description': """
        **Example**: Service A sets timeout=30s, Service B expects timeout=60s.
        
        **Real Impact**: 60% of config CSI failures involve silently ignored configs.
        
        **Root Cause**: Configs span multiple systems, interactions are opaque.
        """,
        'systems': 'ServiceA → ServiceB'
    },
    
    FailureType.API_SEMANTIC_VIOLATION: {
        'title': '🔌 API Semantic Violation',
        'description': """
        **Example**: Upstream assumes synchronous API, Downstream is async.
        
        **Real Impact**: Thread-safety, ordering, synchrony mismatches.
        
        **Root Cause**: Implicit semantics not machine-checkable.
        """,
        'systems': 'Upstream → Downstream'
    },
    
    FailureType.TYPE_CONFUSION: {
        'title': '🔢 Type Confusion',
        'description': """
        **Example**: System A uses `float`, System B expects `decimal` for money.
        
        **Real Impact**: Precision loss, wrong calculations.
        
        **Root Cause**: Type systems don't cross system boundaries.
        """,
        'systems': 'SystemA → SystemB'
    }
}

selected_scenario = st.selectbox(
    "Failure Scenario",
    list(FailureType),
    format_func=lambda x: scenario_descriptions[x]['title']
)

col1, col2 = st.columns([2, 1])

with col1:
    st.info(scenario_descriptions[selected_scenario]['description'])

with col2:
    st.metric("Systems Involved", scenario_descriptions[selected_scenario]['systems'])

# Simulation parameters
st.subheader("⚙️ Simulation Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    duration = st.slider("Simulation Duration (s)", 10, 120, 60)

with col2:
    request_rate = st.slider("Request Rate (req/s)", 1, 50, 10)

with col3:
    st.metric("Expected Requests", int(duration * request_rate))

# Run simulation
if st.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Simulating cross-system interactions..."):
        stats, pipeline = run_csi_simulation(selected_scenario, duration, request_rate)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", stats['total_requests'])
        
        with col2:
            st.metric("Successful", stats['successful_requests'],
                     delta=f"{stats['success_rate']:.1f}%")
        
        with col3:
            st.metric("Failed", stats['failed_requests'],
                     delta=f"-{100-stats['success_rate']:.1f}%")
        
        with col4:
            st.metric("CSI Failures", stats['csi_failures'],
                     help="Failures due to system interaction discrepancies")
        
        # Plot results
        fig = plot_csi_results(stats, selected_scenario)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        st.subheader("🔍 Failure Analysis")
        
        csi_rate = (stats['csi_failures'] / stats['failed_requests'] * 100) if stats['failed_requests'] > 0 else 0
        
        if stats['csi_failures'] > 0:
            st.error(f"""
            ❌ **CSI Failures Detected!**
            
            **{stats['csi_failures']}** failures ({csi_rate:.1f}% of all failures) were caused by 
            cross-system interaction discrepancies.
            
            **Key Characteristics**:
            - Both systems functioned correctly in isolation
            - Failure emerged only through interaction
            - Could not be detected by unit testing either system alone
            - Manifested as crashes/rejections (74% in real systems)
            
            **Root Cause for {selected_scenario.value}**:
            {scenario_descriptions[selected_scenario]['description']}
            
            **Why This Happens**:
            - No cross-system integration testing
            - Schema/config specifications not machine-checkable
            - Implicit assumptions about interaction semantics
            - Systems evolve independently (version skew)
            """)
        else:
            st.success("✅ No CSI failures detected in this run (may need more requests)")
        
        # System-level breakdown
        if stats['system_failures']:
            st.subheader("📊 System-Level Breakdown")
            
            failure_df = pd.DataFrame([
                {
                    'System': system,
                    'Failures': count,
                    'Percentage': f"{count/stats['failed_requests']*100:.1f}%" if stats['failed_requests'] > 0 else "0%"
                }
                for system, count in stats['system_failures'].items()
            ])
            
            st.dataframe(failure_df, use_container_width=True)
            
            # Identify the bottleneck
            if len(stats['system_failures']) > 0:
                bottleneck = max(stats['system_failures'].items(), key=lambda x: x[1])
                st.warning(f"""
                ⚠️ **Bottleneck Identified**: {bottleneck[0]}
                
                This system is the primary failure point with {bottleneck[1]} failures.
                
                **Typical Fix Location**: In real systems, 69% of CSI fixes are applied to 
                the **upstream system** (the one calling the API), often in dedicated 
                "connector" modules.
                """)
        
        # Prevention strategies
        with st.expander("🛡️ How to Prevent This CSI Failure"):
            if selected_scenario == FailureType.METADATA_MISMATCH:
                st.markdown("""
                ### Prevention Strategies
                
                1. **Unified Schema Registry**
                   ```python
                   # Define schema once, use everywhere
                   schema = SchemaRegistry.get("user_event_v1")
                   hive_writer.use_schema(schema)
                   spark_reader.use_schema(schema)
                   ```
                
                2. **Schema Evolution Rules**
                   - Version all schemas
                   - Enforce backward compatibility
                   - Use Avro/Protobuf for strong typing
                
                3. **Cross-System Integration Tests**
                   ```python
                   def test_hive_spark_compatibility():
                       data = hive.write(sample_data)
                       result = spark.read(data)
                       assert result == sample_data
                   ```
                
                4. **Write-Read Testing**
                   - Write with System A
                   - Read with System B
                   - Verify data integrity
                """)
            
            elif selected_scenario == FailureType.SCHEMA_CONFLICT:
                st.markdown("""
                ### Prevention Strategies
                
                1. **Standardize Serialization**
                   ```python
                   # Use unified serialization library
                   data = StandardSerializer.serialize(obj, encoding='utf-8')
                   # All systems use same serializer
                   ```
                
                2. **Encoding Declaration**
                   ```python
                   # Explicitly declare in metadata
                   message.headers['Content-Encoding'] = 'utf-8'
                   message.headers['Charset'] = 'utf-8'
                   ```
                
                3. **Validation at Boundaries**
                   ```python
                   def consume_message(msg):
                       if msg.encoding != 'utf-8':
                           raise EncodingMismatchError()
                   ```
                """)
            
            elif selected_scenario == FailureType.CONFIG_INCOHERENCE:
                st.markdown("""
                ### Prevention Strategies
                
                1. **Centralized Configuration**
                   ```python
                   # Single source of truth
                   config = ConfigService.get_config("pipeline_v1")
                   serviceA.apply(config)
                   serviceB.apply(config)
                   ```
                
                2. **Configuration Validation**
                   ```python
                   # Validate consistency across systems
                   def validate_cross_system_config(systems):
                       for key in shared_configs:
                           values = [s.get_config(key) for s in systems]
                           assert len(set(values)) == 1
                   ```
                
                3. **Contract Testing**
                   - Document expected configs
                   - Test that systems respect contracts
                   - Fail fast on mismatch
                """)
            
            elif selected_scenario == FailureType.API_SEMANTIC_VIOLATION:
                st.markdown("""
                ### Prevention Strategies
                
                1. **Explicit API Contracts**
                   ```python
                   @api_contract(
                       synchronous=True,
                       thread_safe=False,
                       ordered=True
                   )
                   def process_batch(items):
                       pass
                   ```
                
                2. **Machine-Checkable Specifications**
                   - Use OpenAPI/gRPC for typed APIs
                   - Specify concurrency semantics
                   - Document ordering requirements
                
                3. **Cross-System API Testing**
                   ```python
                   def test_api_semantics():
                       # Test synchrony
                       result = upstream.call(request)
                       assert result is not Future
                       
                       # Test ordering
                       results = [call(i) for i in range(10)]
                       assert results == sorted(results)
                   ```
                """)
            
            else:  # TYPE_CONFUSION
                st.markdown("""
                ### Prevention Strategies
                
                1. **Strong Typing Across Systems**
                   ```python
                   # Use Protocol Buffers or similar
                   message Amount {
                       int64 value = 1;    # Always cents
                       string currency = 2;
                   }
                   ```
                
                2. **Explicit Type Conversions**
                   ```python
                   # Make conversions explicit and validated
                   def to_decimal(float_val):
                       decimal_val = Decimal(str(float_val))
                       if abs(decimal_val - float_val) > 0.0001:
                           raise PrecisionLossError()
                       return decimal_val
                   ```
                
                3. **Type Safety Tests**
                   ```python
                   def test_amount_precision():
                       amount = 123.45
                       a_val = systemA.process(amount)
                       b_val = systemB.process(amount)
                       assert abs(a_val - b_val) < 0.001
                   ```
                """)

# Educational sections
with st.expander("📚 Understanding CSI Failures"):
    st.markdown("""
    ### What Makes CSI Failures Different?
    
    **Not Dependency Failures**:
    - Dependency failure: Downstream is unavailable
    - CSI failure: Downstream is available but interaction fails
    
    **Not Library Bugs**:
    - Library bug: Single address space, well-tested
    - CSI failure: Cross-system boundary, expensive to test
    
    **Not Component Failures**:
    - Component failure: System fails in isolation
    - CSI failure: Both systems work alone, fail together
    
    ### Research Findings (EuroSys 2023)
    
    **Prevalence**:
    - 20% of catastrophic cloud incidents (AWS, Azure, Google)
    - 37% of failures in 7 major open-source systems
    - Incidents lasted 10 minutes to 19 hours (median: 106 min)
    
    **Failure Planes**:
    - Data plane: 51% (metadata inconsistencies)
    - Management plane: 32% (configuration issues)
    - Control plane: 17% (API semantics)
    
    **Manifestation**:
    - 74% result in crashes/hangs
    - Existing fault tolerance doesn't protect interactions
    - Cross-system interactions are single points of failure
    
    **Fix Patterns**:
    - 69% fixed in upstream system (connector code)
    - 40% of fixes are workarounds (condition checking)
    - Only 60% address root interaction issues
    
    ### Why Testing Doesn't Catch CSI Failures
    
    **Unit Tests**: Don't cover cross-system interaction
    
    **Integration Tests**: Only 6% test across systems (Spark study)
    
    **Production Testing**: Systems co-deployed in specific versions
    
    **Root Cause**: No standard practice for cross-system testing
    
    ### The Growing Problem
    
    Modern trends make CSI failures more likely:
    - **Microservices**: More system boundaries
    - **Sky Computing**: Cross-cloud orchestration
    - **Hybrid Cloud**: On-prem + cloud integration
    - **Serverless**: More fine-grained composition
    
    ### Prevention Requires New Approaches
    
    1. **Cross-System Testing**
       - Test systems together, not just in isolation
       - Use production configurations
       - Test version combinations
    
    2. **Schema/Config Enforcement**
       - Machine-checkable specifications
       - Automated validation
       - Break builds on incompatibility
    
    3. **Connector Module Focus**
       - 86% of fixes in connectors (<5% of codebase)
       - Heavy testing of integration points
       - Treat connectors as critical path
    
    4. **Change Analysis**
       - Assess cross-system impact
       - Version compatibility matrix
       - Staged rollouts with both systems
    """)

with st.expander("🎯 Real-World Examples"):
    st.markdown("""
    ### Case Study 1: Spark-Hive Schema Mismatch
    
    **Problem**: Spark wrote data using its schema, Hive couldn't read it back.
    
    **Root Cause**: Different timestamp type representations.
    
    **Impact**: 15 distinct discrepancies found in simple write-read testing.
    
    **Fix**: Standardized schema in Parquet format, added cross-system tests.
    
    ### Case Study 2: Kafka-Flink Configuration
    
    **Problem**: Kafka set `compression.type=lz4`, Flink couldn't decompress.
    
    **Root Cause**: Configuration silently ignored in Flink.
    
    **Impact**: Data corruption, silent failures.
    
    **Fix**: Added config validation, fail fast on unknown configs.
    
    ### Case Study 3: Mars Climate Orbiter (Classic CSI)
    
    **Problem**: One system used metric, another used US customary units.
    
    **Root Cause**: No unit enforcement across teams.
    
    **Impact**: $125 million mission failure.
    
    **Lesson**: Even simple type mismatches can be catastrophic.
    
    ### Case Study 4: AWS Kinesis Outage (2020)
    
    **Problem**: Metadata mismatch between control and data plane.
    
    **Root Cause**: Schema evolution broke interaction.
    
    **Impact**: 7+ hour outage, cascading failures.
    
    **Fix**: Improved cross-plane testing, schema versioning.
    
    ### Industry Response
    
    **Growing Recognition**: CSI failures now part of reliability engineering.
    
    **New Tools**:
    - Contract testing frameworks (Pact, Spring Cloud Contract)
    - Schema registries (Confluent Schema Registry)
    - Cross-system fuzzing
    
    **Best Practices**:
    - Write-read testing for every data path
    - Configuration coherence checks
    - Semantic API specifications
    """)
