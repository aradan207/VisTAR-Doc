/**
 * Node Status Classifier
 *
 * Infers a semantic status for each reasoning-tree node based on its
 * tool_calls, position in the tree, and role.  The status drives both
 * the dot colour and the node border/background tint in the UI.
 *
 * Statuses (priority order):
 *   root     – the synthetic user-query node      (indigo)
 *   image    – called image_search                 (purple)
 *   web      – called web_search                   (amber)
 *   search   – called manual_search                (blue)
 *   thinking – has children but no tool_calls      (teal)
 *   final    – terminal leaf (no children)         (green)
 *   done     – fallback / generic completed step   (slate)
 */

/** Colour metadata per status (for legends or tooltips). */
export const STATUS_META = {
  root:     { color: '#6366f1', label: 'User Query' },
  image:    { color: '#a855f7', label: 'Image Retrieval' },
  web:      { color: '#d97706', label: 'Web Search' },
  search:   { color: '#2563eb', label: 'Document Search' },
  thinking: { color: '#0d9488', label: 'Planning' },
  final:    { color: '#16a34a', label: 'Final Answer' },
  done:     { color: '#64748b', label: 'Completed' },
};

/**
 * Classify a single node.
 *
 * @param {object}  node        – the render-state node object
 * @param {boolean} isRoot      – true when the node is the synthetic root
 * @param {boolean} isFinalLeaf – true when the node has no children
 * @returns {string} one of the keys in STATUS_META
 */
export function classifyNode(node, isRoot, isFinalLeaf) {
  if (isRoot) return 'root';

  const tools = Array.isArray(node.tool_calls) ? node.tool_calls : [];
  const toolNames = tools.map((t) => t.tool_name || t.name || '');

  if (toolNames.some((n) => n === 'image_search'))  return 'image';
  if (toolNames.some((n) => n === 'web_search'))     return 'web';
  if (toolNames.some((n) => n === 'manual_search'))  return 'search';

  // Planning / reasoning node: has children but did no tool work itself
  const hasChildren =
    (Array.isArray(node.children) && node.children.length > 0);
  if (hasChildren && tools.length === 0) return 'thinking';

  if (isFinalLeaf) return 'final';

  return 'done';
}
