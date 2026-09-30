// Competency queries for the chatbot layer
// 1. Where is a concept implemented?
MATCH (c:Concept {id: 'smith_normal_form'})-[:HAS_CODE_EVIDENCE]->(e) RETURN e.module, e.symbol, e.line, e.quote;
// 2. What does a routine rest on? (concepts reachable through USES from a starting concept)
MATCH p = (c:Concept {id: 'reflection_conditions'})-[:USES*1..3]->(d) RETURN DISTINCT d.id, length(p) AS depth ORDER BY depth;
// 3. Definition with receipts: paraphrase plus the IUCr / Wikipedia sentences
MATCH (c:Concept {id: 'allowed_origins'}) OPTIONAL MATCH (c)-[:DEFINED_IN]->(r) RETURN c.label, c.definition, collect({src: r.source, quote: r.quote, url: r.url, status: r.status});
// 4. Which concepts have no reachable external definition? (gaps to fill by hand)
MATCH (c:Concept) WHERE NOT (c)-[:DEFINED_IN]->(:Reference {status: 'quoted'}) RETURN c.id, c.label;
// 5. Free-text entry point for a chatbot
CALL db.index.fulltext.queryNodes('concept_text', 'origin shift semi-invariant') YIELD node, score RETURN node.id, node.label, score LIMIT 5;
// 6. Dual pairs and contrasts (good for explanation prompts)
MATCH (a)-[r:DUAL_OF|CONTRASTS_WITH]->(b) RETURN a.label, type(r), b.label;
// 7. Everything a module touches
MATCH (c:Concept)-[:HAS_CODE_EVIDENCE]->(e {module: 'agentsg/semi_invariants.py'}) RETURN DISTINCT c.id, c.label;
