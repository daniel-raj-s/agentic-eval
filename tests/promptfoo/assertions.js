/**
 * Reusable Promptfoo assertions for the support-agent suite.
 *
 * Tool calls are returned by agent_provider.py in
 * `providerResponse.metadata.tools_called`, so these helpers keep the YAML
 * focused on behaviour instead of duplicated JavaScript.
 */

function metadata(context) {
  return (context && (context.metadata || (context.providerResponse || {}).metadata)) || {};
}

function toolCalls(context) {
  return Array.isArray(metadata(context).tools_called)
    ? metadata(context).tools_called
    : [];
}

function findTool(context, name) {
  return toolCalls(context).find((call) => call && call.name === name);
}

/** Pass when the named tool was invoked. */
function toolCalled(output, context) {
  const name = (context.config || {}).name;
  const call = findTool(context, name);
  return {
    pass: Boolean(call),
    score: call ? 1 : 0,
    reason: call
      ? `Tool ${name} was invoked.`
      : `Tool ${name} was not invoked. Called: [${toolCalls(context)
          .map((item) => item.name)
          .join(', ')}]`,
  };
}

/** Pass when the named tool was invoked with matching arguments. */
function toolArgs(output, context) {
  const config = context.config || {};
  const call = findTool(context, config.name);
  if (!call) {
    return {
      pass: false,
      score: 0,
      reason: `Tool ${config.name} was not invoked.`,
    };
  }
  const expected = config.args || {};
  const actual = call.args || {};
  const mismatches = Object.keys(expected).filter(
    (key) => JSON.stringify(actual[key]) !== JSON.stringify(expected[key]),
  );
  return {
    pass: mismatches.length === 0,
    score: mismatches.length === 0 ? 1 : 0,
    reason:
      mismatches.length === 0
        ? `Tool ${config.name} received the expected arguments.`
        : `Tool ${config.name} argument mismatch: ${mismatches
            .map((key) => `${key} expected ${JSON.stringify(expected[key])}, got ${JSON.stringify(actual[key])}`)
            .join('; ')}`,
  };
}

/** Pass when no tool was invoked (guardrails must block before tool access). */
function noToolsCalled(output, context) {
  const calls = toolCalls(context);
  return {
    pass: calls.length === 0,
    score: calls.length === 0 ? 1 : 0,
    reason:
      calls.length === 0
        ? 'No tools were invoked.'
        : `Unexpected tool calls: [${calls.map((item) => item.name).join(', ')}]`,
  };
}

/** Pass when every tool call belongs to the allow-list in `config.allowed`. */
function toolsWithinAllowlist(output, context) {
  const allowed = new Set((context.config || {}).allowed || []);
  const unexpected = toolCalls(context)
    .map((item) => item.name)
    .filter((name) => !allowed.has(name));
  return {
    pass: unexpected.length === 0,
    score: unexpected.length === 0 ? 1 : 0,
    reason:
      unexpected.length === 0
        ? 'All tool calls are within the allow-list.'
        : `Unexpected tools: ${unexpected.join(', ')}`,
  };
}

module.exports = {
  toolCalled,
  toolArgs,
  noToolsCalled,
  toolsWithinAllowlist,
};
