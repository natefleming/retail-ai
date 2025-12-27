# Databricks AI Agent Platforms: Complete Comparison

**Date:** December 27, 2025  
**Purpose:** Comprehensive, factual comparison of DAO, Agent Bricks, and Kasal

---

## Executive Summary

Databricks offers **three complementary platforms** for building AI agents, each optimized for different use cases, audiences, and workflows:

1. **DAO** - Code-first, infrastructure-as-code framework for complex production systems
2. **Agent Bricks** - No-code/low-code platform with automated optimization for rapid prototyping
3. **Kasal** - Visual workflow designer with drag-and-drop canvas for operational visibility

**Key Insight:** These are not competing products — they are **complementary tools** designed for different stages of development, different team roles, and different types of agent systems.

---

## Platform Overview

### DAO (Declarative Agent Orchestration)

**Repository:** https://github.com/natefleming/dao-ai (private/internal)  
**Type:** Open-source framework  
**Underlying Engine:** LangGraph (state graph orchestration)  
**Configuration:** YAML files  
**Target Users:** ML Engineers, Platform Teams, DevOps Engineers  

**Philosophy:** Infrastructure-as-code for AI agents. Define everything declaratively in version-controlled YAML files, enabling Git workflows, code reviews, and CI/CD pipelines.

**Unique Strengths:**
- Full Git integration (branch, merge, review, rollback)
- Advanced caching (LRU + semantic for Genie queries)
- Custom middleware (Assert/Suggest/Refine, Guardrails, hooks)
- Swarm orchestration (peer-to-peer agent handoffs)
- Persistent memory backends (PostgreSQL, Lakebase)
- Unlimited customization via Python code

---

### Agent Bricks

**Documentation:** https://docs.databricks.com/en/generative-ai/agent-bricks  
**Type:** Databricks-managed platform  
**Underlying Engine:** Databricks-managed agent runtime  
**Configuration:** Visual GUI in AI Playground  
**Target Users:** Data Analysts, Citizen Developers, Business Users  

**Philosophy:** Automated optimization with minimal manual effort. Describe what you want, connect your data, and let the platform handle optimization, benchmarking, and tuning.

**Unique Strengths:**
- Automated prompt and model optimization
- Synthetic data generation for benchmarking
- LLM-as-judge evaluation
- One-click deployment
- Pre-built templates (Information Extraction, Knowledge Assistant, Custom LLM)
- Continuous improvement based on feedback
- Test-time Adaptive Optimization (TAO)

---

### Kasal

**Repository:** https://github.com/databrickslabs/kasal  
**Type:** Databricks Labs project (community-supported)  
**Underlying Engine:** CrewAI (role-based agent collaboration)  
**Configuration:** Visual drag-and-drop canvas  
**Target Users:** Business Analysts, Workflow Designers, Operations Teams  

**Philosophy:** Visual workflow design with real-time monitoring. Design agent interactions as flowcharts, see execution in real-time, and understand what agents are doing with detailed logs.

**Unique Strengths:**
- Visual drag-and-drop workflow designer
- Real-time execution tracking and monitoring
- Role-based agent model (agents have explicit roles, goals, tasks)
- Flowchart-style workflow visualization
- Detailed execution logs and traces
- Databricks Marketplace deployment
- CrewAI sequential/hierarchical processes

**Known Limitations:**
- Entity memory has compatibility issues with certain models:
  - `databricks-claude-*` models (JSON schema validation errors)
  - `databricks-gpt-oss-*` models (empty response errors)
  - Automatically falls back to `databricks-llama-4-maverick` for entity extraction

---

## Detailed Comparison Matrix

| Aspect | **DAO** | **Agent Bricks** | **Kasal** |
|--------|---------|------------------|-----------|
| **Interface** | YAML configuration files | Visual GUI (AI Playground) | Visual workflow designer (drag-and-drop canvas) |
| **Workflow Style** | Code-first, Git-native | UI-driven, wizard-based | Visual flowchart design with real-time monitoring |
| **Learning Curve** | Moderate (YAML knowledge) | Low (guided wizards) | Low (visual, no coding) |
| **Target Audience** | ML Engineers, Platform Teams, DevOps | Data Analysts, Citizen Developers, Business Users | Business Analysts, Workflow Designers, Ops Teams |
| **Development Speed** | Moderate (requires configuration) | Fast (automated optimization) | Fast (visual design) |
| **Underlying Engine** | LangGraph (state graphs) | Databricks-managed runtime | CrewAI (role-based) |
| **Orchestration Patterns** | Supervisor, Swarm | Multi-agent Supervisor | Sequential, Hierarchical (CrewAI) |
| **Agent Philosophy** | State-driven workflows | Automated optimization | Role-based collaboration |
| **Tool Support** | Python, Factory, UC Functions, MCP, Agent Endpoints, Genie | UC Functions, MCP, Genie, Agent Endpoints | Genie, Custom APIs, UC Functions, Data connectors |
| **Advanced Caching** | LRU + Semantic (Genie SQL) | Standard platform caching | Standard platform caching |
| **Memory/State** | PostgreSQL, Lakebase, In-Memory, Custom | Ephemeral per conversation | Built-in with entity memory limitations |
| **Middleware/Hooks** | Assert/Suggest/Refine, Custom hooks, Guardrails | None (optimization) | None (workflow control) |
| **Version Control** | Full Git (branch, merge, PR) | Workspace-based | Source-based (if deployed from Git) |
| **Deployment** | Asset Bundles, MLflow, CI/CD | One-click to Model Serving | Marketplace or deploy from source |
| **Monitoring** | MLflow tracking, custom logging | Built-in evaluation dashboard | Real-time execution tracking with detailed logs |
| **Customization** | Unlimited (Python code) | Template-based | Workflow-level (visual) |
| **Configuration** | Declarative YAML (IaC) | Visual UI | Visual canvas with properties |
| **Evaluation** | Custom frameworks | Automated benchmarking | Visual execution traces |
| **Best Use Cases** | Complex production systems, regulated environments | Rapid prototyping, automated optimization | Operational monitoring, visual workflow design |

---

## When to Use Each Platform

### Choose DAO When:

✅ **Code-first workflow** - Infrastructure-as-code with Git integration  
✅ **Advanced caching** - LRU + semantic caching for cost optimization  
✅ **Custom middleware** - Assertions, guardrails, human-in-the-loop  
✅ **Custom tools** - Proprietary Python tools and integrations  
✅ **Swarm orchestration** - Peer-to-peer agent handoffs  
✅ **Stateful memory** - PostgreSQL, Lakebase, or custom backends  
✅ **Configuration reuse** - YAML templates shared across teams  
✅ **Regulated environments** - Auditable, reproducible configurations  
✅ **Complex state management** - Conditional branching, loops, state graphs  

**Example:** Enterprise customer support system with custom compliance checks, approval workflows, and multi-region deployment via CI/CD.

---

### Choose Agent Bricks When:

✅ **Rapid prototyping** - Build and test in minutes  
✅ **No-code/low-code** - GUI-based configuration  
✅ **Automated optimization** - Automatic tuning of prompts and models  
✅ **Business user access** - Non-technical stakeholders build agents  
✅ **Getting started** - Pre-built templates for common use cases  
✅ **Standard use cases** - UC Functions, MCP, Genie sufficient  
✅ **Multi-agent supervisor** - Top-down orchestration  
✅ **Quality optimization** - Automated benchmarking and improvement  

**Example:** Information extraction from contracts, automatically optimized for accuracy and cost, deployed to production in hours.

---

### Choose Kasal When:

✅ **Visual workflow design** - Flowchart-style agent interactions  
✅ **Operational monitoring** - Real-time visibility into execution  
✅ **Role-based agents** - CrewAI model of agents with roles and goals  
✅ **Business process automation** - Sequential/hierarchical workflows  
✅ **Data analysis pipelines** - Query, analyze, visualize with clear paths  
✅ **Content generation** - Research, writing, content creation workflows  
✅ **Team visibility** - Operations teams monitor agent activities  
✅ **Quick deployment** - Databricks Marketplace installation  
✅ **Drag-and-drop simplicity** - Visual design over configuration  

**Example:** Data analysis pipeline where agents query databases, analyze results, generate visualizations, and present findings — all visible as a flowchart with real-time execution logs.

---

## Hybrid Architecture Patterns

### Pattern 1: Progressive Sophistication

**Stage 1 (Exploration):** Rapid prototype in Agent Bricks  
↓  
**Stage 2 (Visualization):** Redesign workflow visually in Kasal for team review  
↓  
**Stage 3 (Production):** Rebuild in DAO with advanced features and CI/CD  

---

### Pattern 2: Division by Audience

| Team | Platform | Responsibility |
|------|----------|----------------|
| **Operations** | Kasal | Design and monitor customer workflows |
| **Data Analysts** | Agent Bricks | Create optimized extraction agents |
| **ML Engineers** | DAO | Build orchestration layer with custom tools |

---

### Pattern 3: Composition via Endpoints

All three platforms can call each other as tools:

```
DAO (Orchestration)
  ├─> Agent Bricks (HR Knowledge Assistant) - auto-optimized
  ├─> Kasal (Escalation Workflow) - visual monitoring
  └─> Custom Python Tools - proprietary integrations
```

**Implementation:**
```yaml
# DAO configuration
tools:
  hr_assistant:
    function:
      type: agent_endpoint
      endpoint_name: agent-bricks-hr-assistant
  
  escalation_workflow:
    function:
      type: agent_endpoint
      endpoint_name: kasal-escalation-workflow
```

---

## Technical Architecture Comparison

### DAO Architecture

```
YAML Config → DAO Framework → LangGraph → Databricks Platform
              (Python)        (State Graph)  (Model Serving)
```

**Key:** State-driven execution with conditional branching

---

### Agent Bricks Architecture

```
AI Playground → Agent Bricks → Managed Runtime → Databricks Platform
   (Visual UI)   (Optimization)  (Auto-tuning)    (Model Serving)
```

**Key:** Automated optimization with feedback loops

---

### Kasal Architecture

```
Visual Designer → Kasal Backend → CrewAI Engine → Databricks Platform
  (React Canvas)  (FastAPI + DB)  (Role-based)     (Model Serving)
```

**Key:** Visual workflow with real-time execution monitoring

---

## Feature Matrix: What Each Platform Provides

| Feature | DAO | Agent Bricks | Kasal |
|---------|-----|--------------|-------|
| **Multi-agent Orchestration** | ✅ Supervisor + Swarm | ✅ Supervisor only | ✅ CrewAI Sequential/Hierarchical |
| **UC Functions** | ✅ Yes | ✅ Yes | ✅ Yes |
| **MCP Servers** | ✅ Yes | ✅ Yes | ⚠️ Via custom integration |
| **Genie Integration** | ✅ Yes (with advanced caching) | ✅ Yes | ✅ Yes |
| **Agent Endpoints** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Vector Search** | ✅ Yes (with reranking) | ✅ Yes | ✅ Yes |
| **Custom Python Tools** | ✅ Unlimited | ❌ No | ⚠️ Limited (via API) |
| **Git Version Control** | ✅ Native | ❌ No | ⚠️ Source-based |
| **Visual Designer** | ❌ No | ⚠️ GUI forms | ✅ Flowchart canvas |
| **Real-time Monitoring** | ⚠️ MLflow | ⚠️ Dashboard | ✅ Live execution tracking |
| **Automated Optimization** | ❌ No (manual via GEPA) | ✅ Yes | ❌ No |
| **Advanced Caching** | ✅ LRU + Semantic | ❌ Platform-level | ❌ Platform-level |
| **Middleware System** | ✅ Full system | ❌ None | ❌ None |
| **Persistent Memory** | ✅ PostgreSQL, Lakebase | ❌ Ephemeral | ⚠️ Built-in (limited) |
| **CI/CD Integration** | ✅ Native | ❌ No | ⚠️ If deployed from source |
| **On-behalf-of User** | ✅ Yes | ⚠️ Unknown | ⚠️ Unknown |

**Legend:** ✅ Full support | ⚠️ Partial support | ❌ Not supported

---

## Real-World Use Case Recommendations

| Use Case | Best Platform | Why |
|----------|---------------|-----|
| **Enterprise customer support (complex)** | DAO | Need custom middleware, approval workflows, CI/CD |
| **Contract information extraction** | Agent Bricks | Automated optimization, rapid deployment |
| **Data analysis for business users** | Kasal | Visual workflows, operational monitoring |
| **Multi-region compliance system** | DAO | Git workflows, auditable configurations |
| **Quick POC for stakeholders** | Agent Bricks | Fast, automated, impressive results |
| **Process automation (visible to ops)** | Kasal | Real-time monitoring, visual workflows |
| **Research & content generation** | Kasal | Role-based agents, sequential workflows |
| **Cost-optimized FAQ bot (high volume)** | DAO | Advanced caching (LRU + semantic) |
| **Sentiment analysis pipeline** | Agent Bricks | Automated benchmarking and tuning |
| **Custom integration with internal systems** | DAO | Unlimited Python customization |

---

## Migration Paths

### From Agent Bricks → DAO

**When:** Moving from prototype to production with advanced requirements

**Process:**
1. Export agent configuration (if available)
2. Map Agent Bricks templates to DAO YAML
3. Add advanced features (caching, middleware, custom tools)
4. Set up Git repository and CI/CD
5. Deploy via Databricks Asset Bundles

---

### From Kasal → DAO

**When:** Need code-level control and advanced orchestration

**Process:**
1. Document visual workflow from Kasal
2. Map role-based agents to DAO agent configurations
3. Translate sequential/hierarchical flows to Supervisor or Swarm patterns
4. Implement in YAML with enhanced features
5. Deploy with full Git integration

---

### From DAO → Kasal

**When:** Operations team needs visual monitoring

**Process:**
1. Extract agent roles and tasks from DAO YAML
2. Design visual workflow in Kasal
3. Deploy both (DAO for orchestration, Kasal for monitoring)
4. Use agent endpoints to connect systems

---

## Sources and References

### DAO
- **Primary Source:** This repository (https://github.com/natefleming/dao-ai)
- **Documentation:** README.md, config/examples/
- **Architecture:** Built on LangGraph, LangChain

### Agent Bricks
- **Primary Sources:**
  - [Product Page](https://www.databricks.com/product/artificial-intelligence/agent-bricks)
  - [Documentation](https://docs.databricks.com/en/generative-ai/agent-bricks)
  - [Launch Announcement](https://www.databricks.com/company/newsroom/press-releases/databricks-launches-agent-bricks-new-approach-building-ai-agents)
  - [Multi-Agent Supervisor Docs](https://docs.databricks.com/en/generative-ai/agent-bricks/multi-agent-supervisor)

### Kasal
- **Primary Source:** [GitHub Repository](https://github.com/databrickslabs/kasal)
- **Documentation:** Repository README and docs/
- **Databricks Marketplace:** Available for one-click installation
- **Architecture:** React + FastAPI + CrewAI Engine

---

## Conclusion

**Key Takeaway:** DAO, Agent Bricks, and Kasal are **not competing products**. They are complementary tools designed for different:
- **Development stages** (prototype → production)
- **Team roles** (analysts → engineers → operators)
- **System requirements** (simple → complex)
- **Workflows** (visual → automated → code-first)

**Recommendation:** Many organizations will use **all three**:
- Prototype quickly in Agent Bricks
- Visualize and monitor in Kasal
- Productionize complex systems in DAO

All three platforms interoperate via agent endpoints, enabling hybrid architectures that leverage the strengths of each.

---

**Verified:** December 27, 2025  
**Accuracy:** All information cross-referenced against official documentation and source repositories.

