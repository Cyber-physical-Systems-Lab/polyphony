# Review 1 Resolution Plan

This file groups supervisor comments into actionable revision tasks for the thesis report.
Each todo links back to one or more `Comment ID` values in [Fixes_comments.txt](/home/user-rjm/Work/Thesis/rjm/polyphony/Report/Review1/Fixes_comments.txt).
The suggested order is to resolve foundational framing first, then structure and methodology, and finally local editorial fixes.

## 1. High-Priority Structural Revisions

### TODO R1-01: Rebuild the introduction around problem, gap, approach, and thesis scope
- Effort: High
- Linked Comment IDs: 1005, 1006, 1011, 1012
- Affected Sections: Introduction
- What to do: Rewrite the introduction so it moves cleanly from broad context to the specific warehouse coordination problem, explains why the problem matters, states the gap in existing approaches, introduces the thesis approach, and sets up the rest of the report.
- Hint: The supervisor is asking for a clearer narrative spine, not just local wording fixes. The new introduction should make it obvious what problem is being studied, why the chosen testbed is relevant, and how the thesis will answer that problem.
- Done when: A reader can understand the problem, gap, approach, and thesis direction from the introduction alone.
- Progress note: On 2026-04-16, the opening introduction narrative was revised again to preserve a short language-based motivation bridge, then narrow quickly into the warehouse coordination problem, the need for high-level decisions beyond path planning, and the role of the LLM-based control loop. Research-question alignment is still pending.

### TODO R1-02: Make the warehouse setting and testbed concrete
- Effort: High
- Linked Comment IDs: 1006, 1036
- Affected Sections: Introduction, Problem Formulation
- What to do: Add a concise explanation of how the fleet operates in a shared warehouse environment, why path planning alone is insufficient, and how the simulator captures the essential coordination constraints while abstracting lower-level execution details.
- Hint: Mention spatial layout, shared shelves/corridors/chargers, role dependencies, and why task timing and coordination decisions matter beyond motion planning.
- Done when: The reader understands what the environment abstracts and why it is still a meaningful coordination testbed.
- Progress note: On 2026-04-15, the introduction was updated with a more concrete description of shelves, corridors, charging stations, heterogeneous roles, and the Battery-TA-RWARE abstraction.

### TODO R1-03: Rewrite the research questions as measurable questions
- Effort: High
- Linked Comment IDs: 1016, 1027, 1037, 1042, 1044, 1049
- Affected Sections: Introduction, Methodology, Results
- What to do: Replace broad RQs with concrete, testable questions tied to explicit variables and metrics such as deliveries, throughput, calls, invalid actions, coordination failures, latency, or efficiency.
- Hint: The RQs should match what is actually varied in the experiments: model size, planning architecture, prompt representation, and objective progression where relevant.
- Done when: Each RQ can be answered directly with named metrics and corresponding experiment comparisons.
- Status: Substantially completed in the introduction on 2026-04-17.
- Resolution note: The main research question was narrowed, three measurable sub-questions were introduced for model size, planning architecture, and prompt representation, and the introduction now states the primary evaluation criteria explicitly. Further alignment in methodology/results can still be refined, but the supervisor’s core RQ-vagueness concern is now addressed in the report framing.

### TODO R1-04: Decide and document the role of Section 2
- Effort: High
- Linked Comment IDs: 1019, 1020
- Affected Sections: Introduction, Problem Formulation and Objectives
- What to do: Either merge the key parts of Problem Formulation into the introduction, or keep Section 2 but turn it into a short, explicit problem statement that clearly states what is missing in prior work, why it matters, and what exact question the thesis addresses.
- Hint: Do not leave the problem definition split weakly across sections. The reader should not have to infer the gap.
- Done when: The report has one clear location where the problem statement is explicit and testable.
- Status: Completed on 2026-04-17.
- Resolution note: The standalone `Problem Formulation and Objectives` chapter was removed from the rendered report. Its problem-framing role was absorbed into the Introduction, which now provides the explicit gap, motivation, and research-question linkage in one place.

### TODO R1-05: Narrow the thesis scope to a smaller set of primary benefits
- Effort: High
- Linked Comment IDs: 1023, 1024
- Affected Sections: Problem Formulation, Objectives, Results framing
- What to do: Reduce the number of promised benefits and make a clear distinction between primary evaluated claims and secondary observations.
- Hint: Throughput, call-efficiency, and constraint-aware coordination are stronger primary targets than an open-ended list of possible improvements.
- Done when: The report no longer reads as if it is trying to prove too many things at once.
- Status: Completed on 2026-04-17.
- Resolution note: The framing in the Introduction and research questions was narrowed to a smaller set of evaluated outcomes, especially delivery throughput, language-model call efficiency, robustness across scenarios, and observed failure behaviour.

### TODO R1-06: Reframe or remove the Contributions/Purpose/Objectives trio
- Effort: High
- Linked Comment IDs: 1021, 1025, 1026
- Affected Sections: Problem Formulation and Objectives
- What to do: Rework these sections so they support the research questions rather than duplicating methods or listing built components. Remove sections that no longer add value after the RQs are clarified.
- Hint: If contributions remain, they should state what is learned or validated, not just what was implemented.
- Done when: These sections read as research framing, not project-task tracking.
- Status: Completed on 2026-04-17.
- Resolution note: The Contributions section in the Introduction was rewritten into a findings-oriented summary, while the separate Purpose and Objectives subsections were removed from the rendered report because they duplicated the Introduction and Method rather than adding research value.

### TODO R1-07: Explain the objective progression as an experimental design choice
- Effort: High
- Linked Comment IDs: 1028, 1046
- Affected Sections: Methodology, Results
- What to do: Clarify why Objective 1, Objective 2, and Objective 3 exist, what each stage is intended to reveal, and what is held fixed versus changed across the progression.
- Hint: The supervisor sees a risk that this reads as iterative system development. The text should explicitly separate objective change from controller evolution and explain the limits of comparison.
- Done when: The progression looks intentional and interpretable rather than ad hoc.
- Status: Completed on 2026-04-17.
- Resolution note: The Method chapter now contains an `Experiment Families and Objectives` subsection that defines Objective 1, Objective 2, and Objective 3 explicitly, links each stage to what it is intended to reveal, and states that Objective 2 also uses a more mature controller so the progression is not over-interpreted as a pure objective-only comparison.

### TODO R1-08: Create a separate discussion chapter or strongly separated discussion section
- Effort: High
- Linked Comment IDs: 1050
- Affected Sections: Results, Discussion
- What to do: Ensure there is a distinct discussion section that interprets results against the research questions, explains mechanisms, and evaluates limitations instead of mixing raw findings and interpretation.
- Hint: Results should present evidence; discussion should explain what it means and how strong the evidence is.
- Done when: The report clearly distinguishes reporting from interpretation.
- Status: Completed on 2026-04-17.
- Resolution note: The results chapter now includes a dedicated Discussion subsection with separate subsubsections for RQ1, RQ2, and RQ3, followed by broader implications and limitations of evidence, so interpretation is no longer left diffuse across the results text.

## 2. Medium-Priority Content and Method Revisions

### TODO R1-09: Strengthen the literature framing and its link to this thesis
- Effort: Medium
- Linked Comment IDs: 1008, 1009, 1031
- Affected Sections: Introduction, Background and Related Work
- What to do: Reorganize the literature framing so it is not just a generic overview. Group prior work by the aspects relevant to this thesis, and explain how those dimensions connect to the warehouse problem studied here.
- Hint: Use categories that support your design choices, such as coordination structure, information sharing, decision timing, and constraint handling.
- Done when: The literature review helps motivate the thesis instead of feeling detached from it.
- Status: Completed on 2026-04-17.
- Resolution note: The background chapter was reorganized to group prior work by design-relevant dimensions, especially coordination structure, information sharing, decision timing, warehouse task-assignment environments, and grounded language-model planning. The revised text now ties each group more directly to the thesis design choices.

### TODO R1-10: Correct and sharpen the MARL comparison claims
- Effort: Medium
- Linked Comment IDs: 1001, 1010
- Affected Sections: Introduction, Background
- What to do: Rewrite the MARL comparison so it is technically accurate and does not overgeneralize partial observability or controller-design difficulties.
- Hint: Focus on the actual limitation you want to contrast against, such as adaptation cost, objective changes, engineering burden, or grounding in discrete warehouse actions.
- Done when: The MARL comparison is precise, fair, and defensible.
- Status: Completed on 2026-04-17.
- Resolution note: The MARL-related framing in the introduction and abstract was rewritten to avoid overclaiming controller-design issues as MARL properties. The comparison now emphasizes training cost, reward-design dependence, adaptation difficulty, and experimental framing rather than inaccurate generalized claims.

### TODO R1-11: Define the decision interface clearly
- Effort: Medium
- Linked Comment IDs: 1014, 1015, 1017, 1018, 1022, 1029
- Affected Sections: Introduction, Methodology, Problem Formulation
- What to do: Clarify what a “high-level decision” means, when decisions are triggered, what environment feedback is used, and how validation/replanning interact with the LLM output.
- Hint: Be explicit about whether feedback means the post-step consequence, whether decisions are event-driven or step-triggered, and whether chosen actions are only filtered or also semantically validated.
- Done when: The control loop can be understood unambiguously by a reader who has not seen the code.
- Progress note: On 2026-04-17, the introduction and methodology were updated to define high-level decisions as grounded discrete action identifiers, explain that feasibility is enforced both through candidate shaping and post-selection validation, clarify that the controller can use the observed consequence of executed actions after a simulation step, and describe the overall loop as a discrete-time step-synchronous scheme with conditional replanning rather than as continuous control or pure event triggering.

### TODO R1-12: Rewrite the methodology chapter as a justified experimental design
- Effort: Medium
- Linked Comment IDs: 1038, 1041, 1043, 1045
- Affected Sections: Methodology
- What to do: Make the methodology section explain why the study is designed this way, what is fixed and varied, what confounds exist, and how conclusions should be interpreted under those constraints.
- Hint: The supervisor wants justification, not just implementation description. Explicitly discuss fixed seeds, chosen scenarios, evolving prompt/controller logic, and what comparisons are exploratory versus closer to controlled.
- Done when: The methodology reads as an empirical study design, not a build log.
- Status: Completed on 2026-04-17.
- Resolution note: The Method chapter now describes the work explicitly as a comparative empirical study, explains why this design fits the research questions, identifies the main fixed and varied factors, adds interpretation caveats about the validation layer and evolving runner logic, and distinguishes exploratory evidence from stronger within-family comparisons.

### TODO R1-13: Justify scenarios, model size, and prompt format as experimental variables
- Effort: Medium
- Linked Comment IDs: 1037, 1045
- Affected Sections: Background, Methodology
- What to do: Explain why model size, prompt format, and scenario balance/scale are expected to affect constrained multi-agent decision-making and what each scenario family is meant to stress.
- Hint: Tie large or imbalanced scenarios to coordination bottlenecks, and tie prompt/model choices to reliability, feasibility, and computational trade-offs.
- Done when: These choices no longer appear arbitrary.
- Status: Completed on 2026-04-17.
- Resolution note: The introduction and method now both motivate planning architecture, prompt representation, model size, and scenario families as explicit experimental variables. The Scenario and Model Matrix subsection also states what each scenario configuration is intended to stress.

### TODO R1-14: Make the results section answer the controlled comparisons explicitly
- Effort: Medium
- Linked Comment IDs: 1047, 1048
- Affected Sections: Results
- What to do: Present architecture and prompt-format comparisons with direct side-by-side evidence under controlled conditions where other variables are held constant as much as possible.
- Hint: Use aligned tables or tightly paired prose around the same model, scenario, objective, and prompt family so the comparison is immediately visible.
- Done when: The reader can find the architecture and prompt conclusions directly in the results section without relying on later discussion.
- Status: Completed on 2026-04-17.
- Resolution note: The results chapter now includes direct side-by-side evidence tables for the architecture comparison and the prompt-format comparison, and states the main comparative answer explicitly in the results text before later discussion.

### TODO R1-15: Tie ethics and sustainability to concrete system behavior
- Effort: Medium
- Linked Comment IDs: 1030
- Affected Sections: Ethical and Sustainability Considerations
- What to do: Replace generic statements with concrete discussion of safety validation, fallback logic, traceability, model-size/compute trade-offs, and efficiency-related metrics.
- Hint: Anchor the section in what the system actually does: invalid action handling, coordination failure risk, battery safety, and computational cost across models.
- Done when: The section is specific to this thesis and linked to the experimental design.
- Status: Completed on 2026-04-17.
- Resolution note: The ethical and sustainability discussion was moved into Method and rewritten around concrete system properties: grounded action identifiers, candidate shaping, feasibility validation, fallback behavior, logging for traceability, and the trade-off between model capability, compute cost, and coordination efficiency.

## 3. Quick Editorial and Citation Fixes

### TODO R1-16: Expand acronyms at first use
- Effort: Easy
- Linked Comment IDs: 1007, 1013, 1032
- Affected Sections: Introduction, Background
- What to do: Introduce full names before acronym-only references such as LLM, MRS, and MAS.
- Hint: Keep first use explicit and consistent.
- Done when: No acronym appears before its expanded form.
- Status: Completed on 2026-04-17.
- Resolution note: Glossary handling was updated so first use expands consistently in the abstract, Swedish summary, and main text. The report terminology was also normalized from `large language model` to `language model`, and acronyms such as MRS, MAS, and LM now appear with full form on first use.

### TODO R1-17: Clarify abstract wording and thesis summary claims
- Effort: Easy
- Linked Comment IDs: 1000, 1002, 1003, 1004
- Affected Sections: Abstract
- What to do: Tighten the abstract wording so it says exactly what is evaluated and avoids vague or awkward formulations.
- Hint: Focus on what is compared, what is measured, and what the current findings show, without overclaiming.
- Done when: The abstract reads as precise and factual.
- Status: Completed on 2026-04-15.
- Resolution note: The abstract was rewritten to state the compared variables explicitly, name the evaluation criteria, replace vague contribution-style wording with current factual results, and clarify that the thesis evaluates LLM-supported high-level decision-making for heterogeneous multi-robot collaboration in Battery-TA-RWARE.

### TODO R1-18: Improve paragraph flow and sentence linkage
- Effort: Easy
- Linked Comment IDs: 1005, 1011
- Affected Sections: Introduction
- What to do: Merge short fragments where needed and make transitions between sentences and paragraphs more explicit.
- Hint: Add topic sentences and closing links back to the thesis problem.
- Done when: The introduction reads as one coherent argument rather than a set of adjacent observations.
- Progress note: On 2026-04-15, the opening paragraphs of the introduction were rewritten into a more continuous problem-to-approach flow. A full paragraph-flow cleanup in later sections is still pending.

### TODO R1-19: Fix local citation and reference issues while preserving strong definition paragraphs
- Effort: Easy
- Linked Comment IDs: 1033, 1034, 1035
- Affected Sections: Background and Related Work
- What to do: Resolve ambiguous references such as “Zhang et al.?”, add missing citations where definitions or claims need support, and preserve the terminology explanations that were already working well.
- Hint: If a statement is definitional or comparative, make sure the source clearly supports that exact claim. Keep the existing clear distinctions between terms rather than rewriting them unnecessarily.
- Done when: All flagged citation ambiguities are resolved and the good terminology explanations remain intact.
- Status: Completed on 2026-04-17.
- Resolution note: Ambiguous local references in the background chapter were clarified, including the Zhang et al. example, while the strong terminology-definition paragraph was preserved. Where support was weaker than needed, the wording was softened rather than left as an over-strong unsupported claim.

### TODO R1-20: Decide whether to add one explanatory visual earlier
- Effort: Easy
- Linked Comment IDs: 1039, 1040
- Affected Sections: Introduction or Methodology
- What to do: Consider moving or adding a visual that helps the reader understand the warehouse/testbed and battery-aware setting earlier in the report.
- Hint: One concise system illustration can reduce explanation load in the introduction.
- Done when: Either a visual is added or moved with a clear purpose, or the report explicitly remains text-only by choice.
- Status: Completed on 2026-04-17.
- Resolution note: The warehouse-setting figure was moved earlier into the Introduction, and an additional control-loop schematic was added in Method to clarify the staged interaction between prompting, validation, execution, and replanning.

## 4. Execution Order

Use this order when resolving the comments:

1. R1-01 to R1-07
2. R1-11 to R1-14
3. R1-09 to R1-10 and R1-15
4. R1-16 to R1-20
5. Final consistency pass:
   - verify every `Comment ID` from `1000` to `1050` is covered by at least one todo
   - update `Fixes_comments.txt` entries using the todo IDs and actual changes made
   - rebuild the report and check that section names, RQs, and results/discussion alignment are consistent

## 5. Resolution Tracking Notes

- One supervisor comment may map to more than one todo.
- One todo may resolve multiple related comments.
- When a todo is completed, the corresponding `Fix Comment:` field in [Fixes_comments.txt](/home/user-rjm/Work/Thesis/rjm/polyphony/Report/Review1/Fixes_comments.txt) should be filled with the actual resolution and the matching todo ID, for example `Resolved via R1-03 and R1-06`.
- `Fixes_comments.txt` remains the comment-by-comment source of truth, while this document is the grouped working plan.
