/** Physical SPARTAN filter ID; the final suffix identifies a replicate. */
export const baseFilterId = (id: string) => id.replace(/^([A-Z]{4}-\d{4})-\d+$/, '$1')
