# Error Analysis

This file walks through real examples of when each search system wins and
loses, and explains *why* in plain terms.

**Quick recap of the test:** for each of 412 papers, search using that
paper's *title* and check where its abstract shows up in the results. A
rank of 1 means it was the top result. "Not in top 20" means the system
never surfaced it at all.

## Where keyword search (BM25) wins

| Search query (a paper's title) | BM25 found it at rank | Semantic found it at rank |
|---|---|---|
| "Dominating vs. Dominated: Generative Collapse in Diffusion Models" | 1 | not in top 20 |
| "Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations" | 1 | not in top 20 |
| "GENIUS: An Agentic AI Framework for Autonomous Design and Execution of Simulation Protocols" | 1 | not in top 20 |
| "Hard Negative Sample-Augmented DPO Post-Training for Small Language Models" | 3 | not in top 20 |
| "EMAG: Self-Rectifying Diffusion Sampling with Exponential Moving Average Guidance" | 1 | 19 |

**Why this happens:** every one of these titles has an unusual, one-of-a-kind
word, such as a made-up acronym (`GENIUS`, `EMAG`), a named technique
(`DPO`), or a quirky phrase (`Tame Your LLM Demons`), and that same word
also shows up in the paper's abstract. Keyword search just checks "does this
rare word appear in the document?" and gets an easy, unambiguous yes.

Semantic search doesn't look for exact words. It looks for the general
*topic* of the title (e.g. "diffusion models," "agentic AI," "reducing
hallucinations"). The problem is dozens of other papers share that same
general topic, so the one correct paper gets lost in a crowd of similar-
sounding papers and never makes it into the top 20 results.

**In short:** if a title contains a rare, distinctive word that's repeated in
the abstract, keyword search wins easily.

## Where semantic search wins

| Search query (a paper's title) | BM25 found it at rank | Semantic found it at rank |
|---|---|---|
| "Towards Efficient Agents: A Co-Design of Inference Architecture and System" | not in top 20 | 1 |
| "Non-Linear Counterfactual Aggregate Optimization" | not in top 20 | 1 |
| "An item is worth one token in Multimodal Large Language Models-based Sequential Recommendation" | not in top 20 | 3 |
| "From Word to World: Can Large Language Models be Implicit Text-based World Models?" | 17 | 1 |
| "Architectures for Building Agentic AI" | not in top 20 | 6 |

**Why this happens:** these titles are either generic ("Architectures for
Building Agentic AI") or written as wordplay rather than plain description
("From Word to World," "An item is worth one token," which riffs on "a
picture is worth a thousand words"). Keyword search has no rare words to
latch onto, so it either finds nothing useful, or it accidentally matches a
completely different, unrelated paper that happens to share common words
like "agents" or "word."

Semantic search doesn't care that the title is worded differently from the
abstract. It understands that both are describing the same underlying idea,
even with zero words in common, and ranks the correct paper first or close
to it.

**In short:** if a title is vague, generic, or metaphorical rather than
using the paper's technical vocabulary directly, semantic search wins
easily.

## The takeaway

Each system fails in a predictable, opposite way:

- **Keyword search (BM25)** is precise when the exact words match, but
  breaks when they don't.
- **Semantic search** understands meaning even when the words are
  completely different, but can lose track of one specific paper when many
  papers share the same general topic.

Neither system is strictly better. They make different kinds of mistakes.
That's exactly why combining both signals (a "hybrid" approach) is a natural
next step: use keyword matching to catch rare, distinctive terms, and
semantic matching to catch everything else.
