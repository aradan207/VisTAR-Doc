export const providerDefaults = {
  ollama: {
    url: 'http://127.0.0.1:11434',
    model: 'hf.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M',
  },
};

export function initProviderDefaults(elements) {
  // Set default Ollama values if not already set
  const defaults = providerDefaults.ollama;
  if (!elements.providerUrl.value) {
    elements.providerUrl.value = defaults.url;
  }
  if (!elements.modelName.value) {
    elements.modelName.value = defaults.model;
  }
}
