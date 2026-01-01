# 07. Human-in-the-Loop (HITL)

**Approval workflows for sensitive operations**

Pause agent execution to get human approval before executing critical actions like deletions, external communications, or financial transactions.

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| `human_in_the_loop.yaml` | Tool approval workflows | Sensitive operations requiring review |

## What You'll Learn

- **HITL Configuration** - How to require approval for specific tools
- **Approval Workflows** - Configure approve, edit, and reject decisions
- **State Management** - Preserve context across approval cycles
- **Review Prompts** - Provide context to human reviewers

## Quick Start

```bash
dao-ai chat -c config/examples/07_human_in_the_loop/human_in_the_loop.yaml
```

Request a sensitive action - the agent will pause and wait for your approval before proceeding.

## Configuration Pattern

```yaml
tools:
  send_email_tool:
    name: send_email
    function:
      type: python
      name: my_package.send_email_function
      human_in_the_loop:
        review_prompt: "Review this email before sending"
        allowed_decisions:
          - approve  # Execute with original parameters
          - edit     # Modify parameters before execution
          - reject   # Cancel the operation
```

## Decision Types

### Approve
Execute the tool with the original parameters provided by the agent.

### Edit
Modify the tool parameters before execution. Useful when the agent got most of it right but needs minor adjustments.

### Reject
Cancel the operation entirely. The agent will be notified and can try an alternative approach.

## Prerequisites

- ✅ Checkpointer configured (PostgreSQL or Lakebase) for state persistence
- ✅ MLflow for managing interrupts
- ✅ Approval UI or API endpoint (optional, can use CLI)

## Use Cases

### Critical Operations
- **Deletions**: Review before removing data
- **Financial**: Approve before spending money
- **External Comms**: Review emails, messages, API calls

### Compliance
- **Audit Trail**: All approvals logged
- **Policy Enforcement**: Require sign-off for regulated actions
- **Risk Management**: Human oversight for high-risk operations

### Quality Control
- **Content Review**: Check generated content before publishing
- **Data Validation**: Verify data transformations
- **Testing**: Manual verification of agent decisions

## Implementation Notes

### State Persistence
HITL requires a checkpointer to maintain state while waiting for approval:

```yaml
memory:
  checkpointer:
    type: postgres  # or lakebase
    connection_string: ${POSTGRES_URL}
```

### Thread Management
Each conversation needs a unique `thread_id` to track approvals:

```python
request = ResponsesAgentRequest(
    input=[Message(role="user", content="Send email to customer")],
    custom_inputs={
        "configurable": {
            "thread_id": "conversation_123",
            "user_id": "reviewer_456"
        }
    }
)
```

### Approval Response Format
When the agent returns an interrupt, respond with:

```python
{
    "action": "approve",  # or "edit" or "reject"
    "modified_params": {...}  # Only needed for "edit"
}
```

## Best Practices

1. **Be Specific**: Write clear review prompts explaining what needs approval
2. **Limit Scope**: Only require HITL for truly sensitive operations
3. **Set Timeouts**: Handle cases where approval never comes
4. **Log Everything**: Maintain audit trail of all approvals
5. **Test Workflows**: Verify approval flows work end-to-end

## Troubleshooting

**"HITL state lost"**
- Ensure checkpointer is configured
- Verify database connectivity
- Check thread_id is preserved across requests

**"Interrupt not raised"**
- Verify tool has `human_in_the_loop` configuration
- Check tool is actually being called by agent
- Review LangGraph interrupt conditions

**"Modified parameters ignored"**
- Ensure you're using "edit" decision, not "approve"
- Verify modified_params matches tool schema
- Check parameter validation logic

## Next Steps

👉 **08_guardrails/** - Automated safety checks  
👉 **13_orchestration/** - Multi-agent workflows with HITL  
👉 **15_complete_applications/** - See HITL in production apps

## Related Documentation

- [HITL Patterns](../../../docs/key-capabilities.md#human-in-the-loop)
- [State Management](../../../docs/key-capabilities.md#state-management)
- [Tool Configuration](../../../docs/configuration-reference.md#tools)
