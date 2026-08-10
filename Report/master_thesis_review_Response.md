Dear Rohit,

Thank you for sending the draft of *Heterogeneous Multi-Robot Collaboration Using Language Models: Exploring Different Sizes and Architectures*. Below are my comments as your supervisor. The core idea — grounding locally hosted language models to a finite, validated action interface in a Battery-TA-RWARE warehouse, and then comparing model sizes, centralised versus per-agent planning, natural-language versus JSON prompting, and objective wordings — is a good, well-scoped master thesis topic. Several parts already read well: the grounded text-in/discrete-action-out control loop is a sound engineering contribution, the related-work chapter is coherently organised, and you are unusually candid about limited runs and the descriptive nature of the results. Keep that candour; it is a strength.

The main problems sit in the second half of the thesis. There is no baseline, so the central question — *to what extent* an LM helps — cannot be answered even in principle from the current data. The experiments are unbalanced and report no uncertainty, so the comparative claims are not auditable. The primary outcome (raw delivery count under an adaptive stopping rule) is not a controlled measure of throughput. Two of the three sub-RQs compare *whole pipelines* rather than the factors they name. Promised failure/robustness results are logged but never reported. And the abstracts and conclusion are not synchronised with the evidence — the Swedish summary even claims a heuristic comparison that appears nowhere in the thesis. None of these require abandoning the framework you built, but they must be fixed before the thesis is defensible.

**Response regarding the baseline:** The thesis is intended as an exploratory feasibility study of whether prompted language models can produce grounded high-level decisions for heterogeneous robot collaboration, and how selected LLM configurations behave within the developed framework. It is not intended to establish that an LLM outperforms non-LLM task-allocation methods. Introducing a baseline would therefore require a separate comparison question and a justified choice among substantially different alternatives, such as a hand-coded heuristic, an optimisation-based task-allocation method, or a particular MARL algorithm and implementation. Selecting, adapting, training, and evaluating such methods fairly would considerably expand the scope of the thesis. The thesis will instead clarify this scope, avoid claims that an LLM “helps” relative to other methods, and revise the main research question and conclusions so that they concern feasibility and the observed differences among the tested LLM-based configurations. Comparative evaluation against non-LLM methods will be stated as future work.

**Response regarding experimental balance and the stopping rule:** The reported model, architecture, and prompt-format comparisons use matched configuration-level observations. Repeated executions with the same objective, architecture, prompt configuration, model, and scenario identifiers are first averaged, after which only configurations with the required comparison counterpart are retained. The prompt comparison contains 144 matched configurations per prompt, and the balanced model comparison contains 48 configurations per model, divided equally between natural-language and JSON prompting. Unequal raw execution totals therefore do not give additional weight to configurations with more repeated runs. The thesis makes descriptive comparisons among the observed configurations rather than inferential claims about a wider population, so confidence intervals are not required to interpret the recorded outcomes. The comparisons remain auditable through the retained run-level results and their corresponding configurations.

**Response regarding raw delivery count:** The termination rule is common and predetermined across all compared configurations. A run does not stop after a fixed total of 500 steps; it stops after 500 consecutive steps without a shelf delivery, while any successful delivery resets the inactivity counter, and completion of all 20 requested shelves also ends the run. This is an intentional stall-detection rule inherited from the simulation design: it allows temporary congestion and complex conflicts to resolve, while terminating executions that have ceased making task progress. Continuing a run when deliveries are still occurring is therefore part of the performance criterion rather than an uncontrolled advantage. Raw delivery count measures how much of the common 20-shelf task is completed before either full completion or sustained inactivity, under the same rule for every configuration.

**Response regarding RQ1 and model size:** Nominal parameter size was the primary model-selection criterion. However, suitable models from a single family were not available across the full evaluated size range. The experiments therefore compare complete off-the-shelf model artifacts spanning different parameter scales, rather than estimating an isolated causal effect of model size. To reflect the actual comparison accurately while preserving its intended focus, RQ1 has been revised to: **“RQ1 (Model Choice Across Parameter Scales): How does grounded collaborative task performance vary among selected locally deployable language models spanning different nominal parameter sizes?”** The corresponding conclusion is that, among the selected models, task performance did not increase monotonically with nominal parameter size.

**Action taken:** The revised RQ1 wording has been applied in the research questions, methodology, results discussion, and conclusion. References to an isolated “effect of model size” have been replaced with observations about performance across the selected complete model artifacts and their nominal parameter scales.

**Response regarding failure and robustness metrics:** Invalid-action and generation-failure fields were implemented primarily as diagnostic instrumentation for debugging and run validation; they were not outcomes associated with a dedicated research question. Abnormal increases were inspected to determine whether they indicated a technical or configuration failure. Runs affected by such failures were excluded, while abnormal model behaviour occurring during otherwise valid executions was retained and reported where relevant. These diagnostics were not intended as a separate comparative robustness study.

**Action taken:** The thesis now states that invalid-action and generation-failure fields are diagnostic run-validation data rather than primary comparative outcomes. It also clarifies that runs are not excluded merely because a model performs poorly or proposes invalid actions; exclusion is limited to technical or configuration failures that make a run inconsistent with the reported protocol. The English and Swedish abstracts have been updated accordingly.

****
I have organised my comments by chapter, and for each section I have tried to give you not just *what* to revise but also *how* to write it. Use the lists and templates as checklists while you rewrite.

*(One housekeeping note first: the task brief named a "mini autonomous cars" platform thesis, but the PDF you submitted is this warehouse-LLM thesis. I have reviewed what you sent. If a different document was intended, let me know — but everything below assumes this is the thesis under examination.)*

---

## 1. Overall framing and research questions

Your main RQ ("to what extent can prompted language models support grounded high-level decision-making for heterogeneous multi-robot collaboration…") plus three sub-RQs (size, architecture, representation) is a sensible decomposition. The problem is that by the time the reader reaches Results and Conclusion, the RQs have effectively disappeared, and — more seriously — three of them are worded as claims the design cannot support.

Before you revise anything else, do this exercise on a single page:

- Write each RQ.
- Under the main RQ, write the **baseline** and the **success threshold** that would let "to what extent" resolve to a positive answer rather than merely "some deliveries occurred". A 20-delivery ceiling is not a threshold.
- Under each sub-RQ, write the *one factor* it claims to isolate, and next to it the list of things that *actually* change together with that factor in your current design. For RQ1 (size): model family, training data, tokenizer, quantization all move with parameter count. For RQ3 (representation): input encoding, output schema, structured-output mode, parser, and token load all move with "JSON". This column is the confound list you must either eliminate or acknowledge.
- Under each, name the experiment and the metric that produce the answer.

This one page is your contract with the reader, and filling it in will immediately show you where the RQ wording outruns the evidence.

**Response:** The baseline concern has been addressed in the overall response above. The main RQ has been revised to remove the phrase “to what extent,” aligning it with the feasibility scope of the thesis. RQ1 has been revised as a comparison of selected models across nominal parameter scales. RQ2 is retained because it directly names the two implemented planning architectures. RQ3 has been revised as a comparison of the tested natural-language and JSON-based prompt configurations. The methodology now states that each prompt configuration includes its input representation, required response format, and associated parsing mechanism. The Conclusion now answers the main RQ and each sub-RQ explicitly.

Two framing fixes for the introduction:

- **State the defensible contribution precisely.** It is *engineering plus preliminary empirical observation*: a grounded, validated LM-to-simulator action interface; two prompt architectures and two encodings within it; a logging/evaluation harness; and descriptive observations. There is no new planning algorithm, no theoretical result, and no validated superiority result — say so explicitly, and distinguish your own modifications from inherited Battery-TA-RWARE/TA-RWARE functionality.
- **Tighten the vocabulary of the claim.** The experiment observes simulator-level task *dependencies*, not communication or negotiation, so the honest phrase is "simulated high-level heterogeneous task assignment", and "controlled comparative experimental design" is too strong while coverage is unequal and factors are coupled. Do not call centrally orchestrated per-agent querying "decentralized".

**Response:** The contribution is already limited to the grounded evaluation framework and descriptive empirical observations; the thesis does not claim a new planning algorithm, a theoretical result, or superiority over non-LLM methods. The term collaboration is retained because AGV and Picker actions are interdependent and both roles are required to complete the shelf-handling task, although direct agent-to-agent messaging and negotiation are not evaluated. The description of the study as a controlled comparison is also retained because each reported comparison uses matched configurations under common experimental conditions, with equal representation of the factor being compared. The intended architecture comparison is centralized joint planning versus shared-context per-agent planning. Four inconsistent uses of “decentralized” in the methodology were terminology remnants and have been corrected.

---

## 2. Methodology (Chapter 3)

Chapter 3 is your strongest technical chapter and the grounded loop is genuinely good. The right test for a methodology chapter is: *could a competent graduate student, given only this chapter, reimplement the framework and get qualitatively similar behaviour?* You are close on the architecture and far on the details — because the prompts, schemas, and thresholds that *are* the method are described narratively rather than specified.

### 2.1 The prompts and schemas are the method — put them on the page

Because this is an LM framework, the prompts are not implementation detail; they are the object of study. A reader who cannot see them cannot tell whether a result is due to the architecture or to prompt engineering. Replace the narrative in §3.4 with a tight, repeated block per prompt stage:

- **Purpose** — one sentence on what this stage decides.
- **Input payload** — a table of fields, types, and where each comes from (the state summariser).
- **Prompt skeleton** — a compact, redacted example of the *actual* prompt.
- **Output schema** — the exact JSON schema or text contract, plus the hold-duration range.
- **Validation and fallback** — what happens on malformed output, timeout, missing agent, out-of-range hold, or a masked action.
- **Role in the comparison** — one sentence linking this stage to centralised vs. per-agent and to natural-language vs. JSON.

Full templates and traces go in the appendix; one representative prompt and one representative output per stage belong in the main text. This will roughly halve the length while doubling the technical content.

**Response:** Appendix A.3 already contains a representative prompt excerpt, and the methodology already describes prompt purpose, structure, configurations, validation, and fallback behaviour. The Prompt Structure section has now been supplemented with a compact representative input and corresponding natural-language and JSON outputs. The methodology also identifies which prompt fields come directly from the simulator, which are derived by the controller, and their basic data types. The request timeout, retry behaviour, hold-duration ranges, and fallback after a failed or incomplete response are now stated explicitly. The complete prompts are generated during each experiment from the current warehouse state. In shared-context experiments, each prompt is tailored to one agent and includes its state, valid actions, and relevant shared information. In centralized experiments, one prompt includes all agents currently requiring decisions. The complete templates remain in the corresponding experiment scripts, while representative examples are included in the report.

### 2.2 Formalise the controller

At the moment the controller is described in prose and flowcharts, and the only numbered equations define reporting metrics. Add:

- a **formal decision-process definition** — agent set, global and per-agent observation, action sets, validity masks, transition timing, battery dynamics, termination conditions;
- **pseudocode** for centralised and per-agent planning, stating explicitly the snapshot semantics and atomicity (in per-agent mode: in what order are agents queried, does each prompt see the same snapshot, and can a later query observe an earlier decision from the same step?);
- a **state machine** for available/busy/held/blocked/resolving/expired/replanning;
- exact rules for mask generation, candidate shaping, fallback, cooldown, target forbidding, and the busy override; and
- the inference-server timeout/retry/crash behaviour.

**Response:** The current methodology already defines the environment state, action space, battery dynamics, termination conditions, control-loop sequence, action masks, candidate shaping, persistence, fallback, conflict handling, busy override, and inference-server failure behaviour. Inspection of the implementation also confirmed that shared-context per-agent queries use a fixed role- and state-based scheduling order. This scheduling determines which eligible agent is queried first but does not select its action. The methodology now documents the ordering, the fixed physical environment snapshot, the visibility of earlier commitments to later prompts, and the point at which the completed action vector is executed. The same scheduling rule is used throughout the shared-context runners and was not investigated as a separate experimental factor. RQ2 is therefore interpreted as a comparison of the complete centralized and shared-context per-agent configurations. A separate formal state-machine diagram and additional pseudocode were not added because they would duplicate the existing numbered control loop, flowchart, and implementation-rule descriptions.

### 2.3 Name the scaffolding as part of the method — and plan to ablate it

This is important. Your system supplies path distances, semantic tags, recommendation flags, explicit role instructions, conflict warnings, A*, collision resolution, Picker staging, action persistence, and a no-op fallback, and it *disables* proactive support. Any of these could dominate both the delivery counts and the apparent architecture/format effects. So in the method, state plainly that the contribution under test is a *co-designed controller*, not an isolated LM planner — and set up §14 of this letter (the ablation) by describing the variants you will compare (recommendation-only, LM-with-full-valid-set, LM-with-shaped-candidates-but-no-recommendation, full pipeline). Also version your simulator modifications: they change the benchmark dynamics, so comparability with published TA-RWARE results is currently uncertain.

**Response:** The methodology already identifies the deterministic components surrounding the language model, including candidate construction, path-based information, validation, action persistence, fallback behaviour, and simulator-managed navigation and collision handling. It also states that the reported behaviour belongs to the complete control framework rather than to the language model in isolation. These components are kept fixed across the reported model, architecture, and prompt-configuration comparisons. Separating their individual contributions would require additional controller variants and a new set of experiments, which is outside the scope of this exploratory study. The proposed recommendation-only and candidate-shaping ablations have therefore been added as future work rather than presented as completed comparisons. The methodology now also distinguishes inherited simulator responsibilities from the modifications introduced for the language-model interface, identifies the source repository and commit, and clarifies that the modified execution dynamics are not treated as directly comparable with previously published TA-RWARE results.

### 2.4 Add a metric→RQ mapping table

Your metric list is fine; what is missing is the mapping. Add a three-column table — *Metric*, *What it measures*, *Which RQ it serves* — and write the "what it measures" entry as a real sentence (e.g. "invalid-action rate measures how often the LM proposes an action outside the current mask, and serves the grounding/robustness contribution"). This is the table you will refer back to throughout Results and Discussion; a metric that cannot be tied to an RQ is a number with no job.

**Response:** A compact metric-mapping table has been added to the methodology. It identifies mean shelf deliveries as the primary outcome for the main RQ and RQ1--RQ3, and identifies LLM calls and deliveries per 1000 calls as secondary planning-overhead and call-normalised measures. Battery-aware productivity is identified separately as part of the exploratory objective-sensitivity analysis rather than as a primary RQ outcome. Invalid-action and generation-failure fields have not been assigned to a robustness RQ because no such RQ is posed in the thesis; they remain diagnostic fields used for run validation.

### 2.5 State the methodological scope and limitations

Say plainly what the method *cannot* answer: one GPU, one inference stack, one simulator family, exact symbolic state (no perception noise), and modified environment dynamics. It is much stronger to bound this here than to be challenged on it at the defence. In particular, "the planner receives exact symbolic ground truth, so the conclusions cannot be transferred to a perception-limited robot without a separate study" is a sentence you want to write yourself.

**Response:** The methodology already identified the simulation-only setting, the local Ollama inference stack, the NVIDIA RTX A4000 platform, and the modified environment dynamics. These boundaries have now been consolidated in the Delimitations subsection. The methodology also states explicitly that the planner receives symbolic simulator state without sensor noise or perception errors, so the study does not establish performance on perception-limited physical robots. Portability across inference servers, hardware platforms, published TA-RWARE results, and other simulator families is likewise identified as outside the evaluated scope.

---

## 3. Experimental setup and protocol (Chapter 3.5–3.7)

The setup is currently the weakest part of the thesis, and the fixes here are the ones that decide whether the whole comparison stands. A setup chapter can be short, but it must be *complete* and *reproducible*, and yours is neither.

### 3.1 Report the run matrix and separate the sources of randomness

Tables 6–8 report no `n` for any cell. Table 9 gives only objective totals (42 / 36 / 90). With 6 models × 6 scenarios = 36 cells before architecture or format, those totals average ~1.2, ~1.0, and ~2.5 runs per cell — and §3.7 says you *added* runs to cover missing combinations, which changes marginal means and creates optional-repetition risk. Write this explicitly instead:

> "We distinguish two sources of variation. *Environment randomness* is controlled through N independent environment seeds, held identical across treatments. *LM sampling randomness* is observed through K repeated runs per seed. The two are reported separately and not pooled as if independent."

Then, in every results statement, say "across N seeds × K runs", never "across NK trials". With a handful of seeds a p-value is not informative — use bootstrap confidence intervals and seed-level paired-difference plots (mutualistic − baseline, or NL − JSON, per seed) so the reader can see whether an effect is consistent across seeds or driven by one or two outliers. And publish a **run manifest**: every attempted / completed / failed / retried / excluded / retained run, with the inclusion rule stated *before* analysis. Right now "completed" and "retained" runs are analysed with no accounting of the rest, which is survivorship bias if completion depends on model or prompt condition.

**Response:** The proposed N-seed × K-repeat formulation does not describe the experimental design used in this thesis. Runs were not collected as a complete crossed set of environment seeds and repeated language-model samples, and presenting them in that form would therefore misrepresent the data. The analysis is descriptive and does not use p-values or make population-level statistical claims.

To prevent configurations with additional executions from receiving greater weight, repeated runs with the same objective, architecture, prompt configuration, model, and scenario are first averaged into one configuration-level observation. Each comparison then retains only observations with a corresponding matched configuration for the factor being compared. The previously reported objective totals of 42, 36, and 90 have consequently been replaced by 72 matched configuration-level observations for each objective.

The statement that Section 3.7 says runs were added to cover missing combinations could not be located in the thesis. Additional executions are not selected according to their performance. Runs that terminate through the specified inactivity rule, including poor-performing runs and runs containing invalid model actions, remain valid observations. Exclusion is limited to technical or configuration failures that make an execution inconsistent with the reported protocol. The retained run-level data and configuration identifiers provide the audit trail for the reported analysis. A complete retrospective manifest of every failed launch or interrupted execution is not claimed.

**Action taken:** The methodology now defines the configuration-level averaging and matching procedure. The objective comparison has been recalculated using equal matched coverage, and the Results chapter reports the number of matched observations rather than the previous unequal run totals. The conclusions remain descriptive and do not make inferential uncertainty claims.

### 3.2 Fix the primary outcome and the stopping rule

There is no fixed horizon: a run continues while occasional deliveries keep resetting a 500-step inactivity counter, and raw total deliveries before stalling are then called throughput. A policy that occasionally makes progress simply gets more steps. This quietly undermines every throughput and efficiency comparison. Two options, either acceptable: (a) re-analyse the logs you *already have* — `deliveries_per_1000_steps`, `total_steps`, `elapsed_seconds`, `steps_to_first_delivery` — or (b) rerun under a common fixed step horizon. Also stop treating "calls" as a common cost unit: one centralised call returns several actions and carries far more tokens than one per-agent call. Report tokens, decisions, latency, and GPU-seconds per delivered shelf instead.

**Response:** Raw delivery count is not intended as a fixed-horizon throughput or time-normalised delivery rate. It measures how much of the common 20-shelf task is completed before either full completion or 500 consecutive steps without a delivery. This termination condition is predetermined and identical across all configurations. A delivery resets the inactivity counter because it demonstrates continued task progress; consequently, a run continues only while it is still completing the task. The outcome is therefore interpreted as delivery count under a common completion-or-stall protocol, not as throughput.

Reanalysing at an arbitrary fixed horizon would answer a different question and could discard valid deliveries from configurations that progress more slowly. Rerunning the complete experiment under a new protocol is outside the scope of the thesis. The current descriptive comparison remains valid for the stated outcome, provided it is not presented as a time- or step-normalised throughput comparison.

The thesis also does not treat an LLM call as a common measure of computational cost. Centralized and shared-context calls differ because a centralized request can return decisions for several agents, whereas shared-context planning queries agents separately. Mean calls therefore measures model-interaction frequency only. Deliveries per 1000 calls is retained as a descriptive call-normalised measure, not as a measure of latency, token cost, energy consumption, or GPU efficiency. In the architecture comparison, mean delivery count remains the primary outcome and call counts are interpreted only as secondary planning overhead.

Tokens, latency, and GPU-seconds were not recorded consistently for the completed experiments and cannot be reconstructed reliably without new measurements. Their evaluation is therefore identified as future work rather than added retrospectively.

**Action taken:** References to raw delivery count as “throughput” have been replaced with “delivery count,” “task completion,” or “delivery performance.” Deliveries per 1000 calls is now described as a call-normalised model-interaction measure rather than computational efficiency or cost. The methodology explicitly explains why call counts are not directly comparable as computational cost between centralized and shared-context planning.

### 3.3 Write the procedure as a recipe

Replace the current setup-like prose with a numbered recipe someone else could replay:

1. Initialise Battery-TA-RWARE with the chosen scenario (map, AGV/Picker counts, chargers).
2. Set the environment seed (from the published seed list).
3. Load the policy (centralised / per-agent / baseline).
4. Initialise Ollama with the exact model digest and decoding settings of §3.5.
5. Run to the fixed horizon.
6. Each step: build the state snapshot and candidate mask.
7. Query the LM stage(s) or the baseline.
8. Parse, validate against the mask; on failure apply the documented fallback.
9. Execute the accepted actions; update battery.
10. Log deliveries, steps, tokens, calls, invalid/fallback counts, failures.
11. Repeat across seeds × runs × scenarios × conditions.
12. Aggregate per cell; report mean, SD, 95% CI, and seed-level paired differences.

Yours is currently missing roughly steps 4, 8, 10, and 12 — the ones that make it reproducible.

**Response:** The methodology already contains a numbered control loop covering state summarisation, candidate-mask construction, prompting, model querying, response parsing, validation, fallback, and action execution. Timeout, retry, malformed-output, missing-action, invalid-action, and fallback behaviour are also documented separately. The suggested procedure additionally assumes a fixed horizon, a non-LLM baseline, a complete seeds × repetitions design, and inferential aggregation; these were not part of the reported study and have therefore not been added as though they were.

A compact run-level procedure has nevertheless been added to the Experimental Protocol subsection. It connects scenario and model selection, environment initialisation, the existing control loop, termination, logging, and configuration-level aggregation. This provides a replayable overview without duplicating the detailed controller description.

Each run records its experimental identifiers and simulator seed together with deliveries, executed steps, LLM calls, invalid or missing model actions, generation failures, and elapsed time. Repeated executions of the same configuration are averaged before matched comparisons, as described in the analysis procedure.

**Action taken:** The Experimental Protocol now contains a concise numbered run procedure and explicitly identifies the principal fields written to the step-level and run-summary logs. The existing detailed control loop and fallback descriptions are retained rather than repeated.

### 3.4 The reproducibility artifacts you must release

State-of-the-thesis reproducibility requires all of this, and none of it is currently in the document: repository URL/DOI, licence, commit; Python / Gym / Battery-TA-RWARE / Ollama / CUDA versions and a lockfile or container; **exact model digests and quantization** (not mutable Ollama display tags); numerical decoding parameters (temperature, top-p/k, max tokens, stop strings, LM seed); all scenario config files; complete NL templates and JSON input/output schemas and parser rules; the actual seed lists; the run manifest above; raw per-run logs, aggregate CSVs, and one command that regenerates every table and figure. The Appendix examples all show seed 0, and §3.5 says decoding is "controlled" without giving the values — those gaps prevent audit, not just polish.

**Response:** The complete artifact package proposed in this comment extends beyond what was recorded during the experimental period, and several items cannot be reconstructed retrospectively without making unsupported claims. The thesis has therefore been updated with the reproducibility information that can be verified.

The source implementation is available at [Cyber-physical-Systems-Lab/polyphony](https://github.com/Cyber-physical-Systems-Lab/polyphony), and the methodology identifies the implementation snapshot by commit hash. The repository contains the project licence, dependency declaration, experiment runners, scenario definitions, prompt construction, output parsing, validation rules, and representative execution commands. A commit hash is used as a single source-version identifier; a complete commit history is not required in the thesis.

The reported runners explicitly set the generation temperature to 0.1 and the maximum generated-token budget to 700. Top-p, top-k, and stop-sequence values are not overridden by the experiment runners and therefore follow the locally installed Ollama model artifacts. No separate language-model sampling seed was configured. The simulator seed is stored with each run.

Exact model digests and complete Python, Ollama, and CUDA environment versions were not recorded consistently when the experiments were executed and cannot be established reliably after the fact. The thesis therefore reports the model tags, inference stack, GPU platform, dependency declaration, and source version that can be verified, without presenting reconstructed values as original experimental metadata.

Raw step-level logs are not included in the Git repository because their size is unsuitable for ordinary source version control. They are retained separately from the source repository. The thesis reports the matched configuration-level data used for analysis and documents how repeated runs were aggregated. A container, public raw-log archive, and automated figure-regeneration pipeline would improve reproducibility but are not necessary contributions to the research questions and are identified as possible archival improvements rather than claimed as completed artifacts.

**Action taken:** The repository URL and source commit are stated in the methodology. The numerical decoding settings and the distinction between explicitly configured parameters and model-artifact defaults have been added. The thesis also clarifies that the source repository contains the implementation and experiment commands but not the large raw result directories.

---

## 4. Results (Chapter 4)

The Results chapter has one job: report what happened, with uncertainty, without interpretation. You do the "without interpretation" part reasonably well in places, but the chapter reports only means, mislabels its main outcome, and answers questions its design cannot isolate. Structure it as: conditions → system-level outcome (deliveries) → efficiency → coordination/failure indicators → a two-paragraph descriptive summary, all in the past tense, with uncertainty on every point estimate.

### 4.1 Report uncertainty and provenance on everything

Add to every table and figure: `n` per cell, SD or IQR, bootstrap 95% CIs, individual-run points, and seed-level paired-difference plots. State, for each table, exactly which runs feed it — the abstract's "call-aware experiments" versus the conclusion's "shelf-delivery objective" currently leave the dataset behind each result ambiguous. Regenerate Figures 7–13 as vector graphics with larger text, accessible markers/hatching (not colour alone), and human-readable scenario names; replace the categorical line graphs in Figures 10–12 (which imply a continuous order that does not exist) with point-range or grouped-bar plots; and write captions that give the aggregation unit, how to read the plot, and a *bounded* conclusion.

**Response:** The requested inferential uncertainty measures are not supported by the experimental design. The data were not collected as a complete balanced set of repeated runs across independent environment seeds. Adding seed-level standard deviations, bootstrap confidence intervals, or paired-seed plots retrospectively would therefore misrepresent the available evidence. The thesis retains a descriptive configuration-level analysis and does not make population-level inferential claims.

The provenance and presentation concerns have been addressed. Repeated executions with identical objective, architecture, prompt configuration, model, and scenario identifiers are first averaged into one configuration-level observation. Only observations with the required matched counterpart are included in each comparison. The Results chapter and the table and figure captions now state the aggregation unit and matched coverage used in each analysis.

Figures 7--13 have been regenerated as native LaTeX vector graphics with larger text, human-readable category names, and patterns in addition to colour. The previous categorical line charts have been replaced with grouped or diverging bar charts. The model ordering is retained to make differences across the selected nominal model sizes easy to inspect, while the bars present the models and scenarios as discrete categories. The captions describe how to read each chart and limit the conclusions to the observed matched configurations. The Conclusion has also been corrected to identify the objective-family configurations used by the revised analyses.

**Action taken:** Matched counts and aggregation units have been added to the Results captions; Figures 7--13 have been replaced with accessible vector charts; and the Results and Conclusion now use consistent descriptions of the analysed datasets. Unsupported inferential uncertainty estimates have not been added.

### 4.2 Name the outcomes accurately, and stop over-reading them

- **Raw delivery count is not throughput.** With a 20-delivery ceiling and no common horizon, the model means of 4.00–10.50 in Table 7 establish that deliveries occur, not that throughput is useful. Rename to "delivery count" or "deliveries per 1000 steps".
- **Architecture cost is under-interpreted.** From the Table 8 means, centralised is *more* call-efficient than shared-context in five of six scenario groups even though shared-context completes more raw deliveries — and calls are not a fair unit anyway. Report tokens and GPU-seconds.
- **RQ3 is not isolated.** §4.2 itself admits the JSON condition changes both input encoding and output constraint, so "natural-language outperformed JSON" is valid only for the two whole pipelines tested, not for representation alone — and the parse/invalid rates that would *explain* the difference are logged but not shown.
- **RQ1 is not a size experiment.** Five of six models are different families; only Llama 1B vs 3B is a within-family scale contrast, and it has no uncertainty. The result supports "nominal size alone did not predict performance", not an estimate of a size effect. ("Mid-sized" for Llama 3B is also misleading in a 1B–14B set — it is the second-smallest.)
- **The battery score is not valid evidence of battery-aware behaviour.** Equation 2 multiplies mean deliveries by mean charger utilization (undefined denominator; scores up to 220.82 show it is not a fraction), which can reward needless charging. Report minimum battery, time below threshold, zero-battery events, and the delivery trade-off instead.

**Response:** The outcome terminology and the limits of the comparisons have been clarified. Raw deliveries are described as delivery count or task completion rather than throughput. LLM calls are treated as model-interaction frequency, not as a common computational-cost unit across the two planning architectures; token use and GPU time were not recorded and have not been reconstructed retrospectively. RQ3 is reported as a comparison of the complete natural-language and JSON-based prompt configurations, and RQ1 is reported as a comparison among selected model artifacts spanning nominal parameter sizes rather than as an isolated parameter-size effect.

The charging-station-utilization denominator has also been defined. It is the occupied charging-station capacity summed across recorded steps divided by the total available charging-station capacity across those steps. The value used in the reported index is expressed in percentage points, so the product with mean deliveries is not itself a percentage and is not bounded by 100. The index was included to examine whether charging-station occupancy coincided with retained delivery performance under different objective prompts. It cannot distinguish necessary charging from unnecessary charging and is not presented as evidence of optimal battery management. The observed effects remain mixed: the Battery objective did not produce a consistent improvement across the evaluated models.

Direct battery outcomes are now also reported for 72 matched Battery and Battery + Shelf configurations per objective. These include mean minimum battery, the fraction of agent-steps below 30, the fraction of agents reaching zero, the fraction of runs reaching zero, and mean deliveries. The Battery objective produced a slightly higher minimum battery and slightly lower zero-battery fractions, but the below-threshold fraction was slightly higher and mean deliveries were lower.

**Action taken:** The remaining throughput and call-efficiency wording has been corrected, the model and prompt comparisons have been bounded to the factors actually observed, and the definition, scale, purpose, and limitations of the battery-aware productivity index have been stated explicitly. A direct matched battery-outcome table has been added using the retained run summaries. No unrecorded token or GPU measurements have been added.

### 4.3 Report the failure evidence you already logged

The abstract, RQs, and contributions all promise failure/robustness behaviour, and Appendix A.4 lists `json_generation_failures`, `llm_missing_or_invalid_actions`, `invalid_action_rate`, `json_failure_rate`, and time-to-first-delivery — yet none appears in Chapter 4. This is missing *analysis of existing instrumentation*, not missing data, and it is one of the cheapest high-value fixes available: the grounding/validation layer is your strongest technical claim, and its error rates are exactly the evidence that would validate it.

**Response:** Failure and invalid-action fields were logged for diagnostic run validation, not as research outcomes. None of the research questions evaluates robustness or failure-generation rates. These fields were used to trace malformed outputs and identify technical or configuration inconsistencies in the experiment data. Poor model performance or invalid-action generation alone was not used to exclude a run. The Abstract, Introduction, Methodology, and Appendix now state this diagnostic role consistently; therefore, no separate failure-rate analysis has been added to the Results chapter.

**Action taken:** The Appendix wording has been clarified so that the logged fields are described as supporting diagnostic tracing rather than as an unreported performance analysis.

### 4.4 Fix the internal inconsistencies

Correct these before an examiner finds them: the Swedish summary claims marked underperformance against a domain heuristic that appears *nowhere* in the method or results (remove it or supply the experiment); the abstract says "highest delivery total" where Table 7 reports a mean; §4.5 says prompt changes "demonstrate" adaptation while §4.6 correctly calls the same evidence descriptive and inconclusive.

**Response:** The three identified inconsistencies are no longer present. The Swedish summary makes no claim about comparison with a domain heuristic, the Abstract does not describe a mean value as the highest delivery total, and the objective-sensitivity discussion consistently describes the observed behaviour as mixed and descriptive rather than as demonstrated adaptation.

**Action taken:** The Swedish summary, Abstract, Results, and Conclusion have been checked for consistency. The remaining Swedish reference to delivery efficiency has also been changed to a call-normalised delivery measure for consistency with the English terminology.

---

## 5. Discussion and research-question answers (Chapter 6)

A discussion is not a re-statement of results; it is where you *answer the RQs* with the evidence, *explain the mechanism*, *bound the claims*, and *connect to the literature*. Structure it as: (1) restatement of findings (descriptive, no new data), (2) one subsection per RQ, (3) mechanism, (4) boundary conditions and negative results, (5) relation to prior work, (6) limitations, (7) threats to validity.

**Response:** The combined Results and Discussion chapter is organized around the three comparison axes associated with RQ1--RQ3, followed by the exploratory objective-sensitivity analysis and a summary of findings. Interpretations are connected to related work where relevant and are bounded by the experimental conditions and methodological delimitations. Separate mechanism and threats-to-validity sections were not added because the available measurements support descriptive interpretation rather than a separate causal analysis.

### 5.1 Answer each RQ explicitly, and bound it

For each RQ, state the answer, cite the specific table/figure, and bound the claim. Concretely:

- **Main RQ ("to what extent"):** currently *insufficiently answered* — you show operability of the combined framework, but without a baseline, a common horizon, uncertainty, and failure rates, "to what extent" has no reference point. Answer it only after the baseline exists (§6 below).
- **RQ1 (size):** answer as a *model-choice* observation, bounded to the tested artifacts, or rerun within one family. Do not claim a scaling effect.
- **RQ2 (architecture):** answer descriptively, and state that the pipelines differ in call count, output cardinality, prompt length, and possibly query ordering, so the delivery difference is not yet attributable to "collaboration".
- **RQ3 (representation):** state that only the whole pipelines are compared, and that input encoding, output schema, and parser are confounded.

**Response:** The main RQ was revised from “to what extent” to a feasibility question and is answered within the tested simulator framework, without claiming superiority over another method. RQ1 is explicitly answered at the level of the selected model artifacts, RQ2 compares the complete centralized and shared-context configurations, and RQ3 compares the complete natural-language and JSON-based prompt configurations. The Results and Conclusion state these bounds and do not present parameter count, architecture, or representation as isolated causal effects.

**Action taken:** The three primary Results subsections are now labelled explicitly as RQ1, RQ2, and RQ3. The Conclusion provides a direct, bounded answer to the main RQ and each sub-RQ.

### 5.2 Explain the mechanism with your behavioural metrics

This is where §3.6 finally pays off — but only once you report the failure and coordination metrics. Use invalid-action rate, fallback rate, parse-failure rate, call counts, and coordination waits to explain *why* a pattern appeared, not just that it did. Do not assert a mechanism you have not instrumented.

**Response:** Invalid-action and generation-failure fields were collected for diagnostic validation rather than as research outcomes, as explained in the methodology and Appendix. They are therefore not used retrospectively to claim an explanatory mechanism. The discussion identifies possible interpretations of the observed patterns, while keeping them explicitly tentative and distinguishing them from measured results.

### 5.3 Be honest about the boundary conditions

If a condition (e.g. picker-scarce or charger-scarce scenarios) is where the approach degrades, discuss it openly and explain it with the data. A thesis that reports where its method fails is stronger, not weaker.

**Response:** Scenario-level results are reported for all six evaluated configurations, including groups with lower delivery performance. The discussion does not infer unmeasured failure mechanisms from those differences. The symbolic-state assumption, modified simulator dynamics, single inference stack and hardware platform, limited initializations, and lack of physical deployment are stated in the methodological delimitations.

### 5.4 Watch the causal language

Your evidence does not isolate a causal mechanism. The strongest you should write is:

> "The observed pattern is consistent with the interpretation that structured grounding prevents out-of-mask execution and that natural-language prompting yields more parseable proposals under the tested conditions."

Not:

> "Natural-language prompting improves collaboration because JSON confuses the model."

The second is a defence trap. With small N, write "the observed pattern is consistent with…", "under the tested conditions…", "the data suggest…" — never "this proves" or "this demonstrates".

**Response:** The Results and Conclusion use descriptive and bounded language, including “suggest,” “possible explanation,” “may,” and “within the tested simulator framework.” Two remaining explanatory statements have been softened further so that they are presented as possible interpretations rather than established causal mechanisms.

---

## 6. The baseline — this is the missing keystone

I am pulling this out on its own because it is the single change that most affects whether the thesis can answer its own main question. Right now you compare only LM configurations against each other, so even a perfect analysis cannot show that the LM adds value over the scaffolding — and because your prompts already expose recommendation fields and explicit instructions, a trivial policy might reproduce much of the behaviour.

Implement, under identical seeds and horizons:

- a **random-valid** baseline,
- a **nearest-feasible / greedy heuristic**,
- a **recommendation-only** policy (follow the candidate recommender with no LM),

and report the incremental effect of adding the LM on top of the deterministic scaffolding, with paired CIs. If the domain-heuristic result claimed in the Swedish summary cannot be recovered from a real experiment, remove that claim immediately. This one addition converts the main RQ from unanswerable to answerable.

**Response:** A non-LM baseline is not required to answer the revised feasibility question, which asks whether prompted language models can produce grounded high-level decisions within the implemented framework. The thesis does not claim that the language model outperforms a heuristic, MARL method, or the deterministic controller components. A fair baseline study would require selecting and justifying additional policies and conducting new matched experiments, which is outside the present scope.

**Action taken:** The unsupported heuristic statement was removed from the Swedish summary. Non-LM baselines and controller-scaffolding ablations are identified as future work, and no comparative-superiority claim is made in the thesis.

---

## 7. Conclusion (Chapter 6/7)

Restructure the conclusion as explicit answers to the main RQ and RQ1–RQ3, followed by a precise statement of contribution status (demonstrated / preliminary / unsupported) and limitations. A good conclusion is a compact, forward-looking closing argument, not a summary:

1. Problem and approach — one paragraph.
2. Principal findings — two or three quantified, bounded takeaways (including the honest "nominal size did not predict performance" and the descriptive architecture/format patterns).
3. Contributions — the grounded interface, the two architectures/encodings, the harness, and the preliminary observations. "An exploratory comparison" is a fine framing; do not overclaim.
4. Limitations — cross-reference, do not repeat.
5. Future work — the genuinely forward items (RAG, MCP tools, explicit V2V communication, edge hardware, more seeds), kept clearly separate from present capabilities. Do not describe proposed future work as implemented.

Note the current conclusion omits an explicit answer to the main RQ and to RQ1 — that is precisely the gap this structure closes.

**Response:** This comment referred to an earlier version. The current Conclusion explicitly answers the main RQ and RQ1--RQ3, distinguishes the primary comparisons from the exploratory objective-sensitivity analysis, and keeps future work separate from present capabilities. It now also includes concise quantitative coverage statements: the architecture result is bounded to six scenario groups and six selected models, and the prompt result is bounded to the matched scenario and model groupings.

---

## 8. Writing, figures, and presentation

A few recurring items to sweep in one pass:

- **Remove draft residue:** the cover still says `UPTEC XX XXXXX`; §1.2 says "current thesis work" and §1.3 "current experimental results"; the report outline on p. 11 omits Chapter 5; and there is an effectively blank p. 45.
- **Reconcile the two abstracts** from the finalised results and limitations, and remove the unsupported heuristic claim from the Swedish summary.
- **Condense the introduction:** pp. 1–9 spend a long time on human-language evolution and basic warehouse exposition before reaching the gap. Move the space to the exact gap, the baseline rationale, and experiment validity.

**Response:** The introduction is intentionally structured to connect language, collaboration, heterogeneous robot coordination, and warehouse task assignment. Familiarity with both language-model concepts and warehouse automation cannot be assumed for the intended interdisciplinary audience. The discussion of language establishes the motivation for treating language as a coordination mechanism, which leads directly to investigating language models as high-level decision-making components. Similarly, the warehouse discussion introduces the heterogeneous roles, task dependencies, and operational constraints needed to understand the experimental environment and research questions. The current progression is therefore retained. A baseline rationale is not expanded here because the thesis does not claim superiority over non-language-model methods; that scope is stated separately.
- **Standardise terminology:** LM/LLM consistently; "delivery count / delivery rate", not "throughput", unless there is a time or step denominator; mask *validity*, not *safety*; define *run / attempt / session / repetition / completed / retained / excluded* once and use them consistently.
- **Verified small corrections:** p. 7 "goods-to-person(GTP)" → add a space; p. 9 remove the doubled "JavaScript Object Notation (JavaScript Object Notation (JSON))"; p. 16 "performance.Xiao et al." missing space; p. 44 `Embedl1 ,` footnote typography; Appendix p. 53 option should read `--prompt_format json`; Reference [7] has an imported `no. null` field.
- **References:** audit non-peer-reviewed sources (arXiv preprints and the Embedl vendor page used for model information), prefer peer-reviewed versions where they exist, and standardise capitalisation, venues, and access dates. Pick British or US English and apply it throughout.
- **AI-tool declaration:** the checklist requires disclosure if generative AI was used for writing, code, translation, or figures. I did not find one — add it if applicable; do not imply use if there was none.

One terminology point worth its own line, because it recurs and matters: your §5 calls mask-filtered actions "unsafe". The documented mask establishes *simulator feasibility* (role and busy-state constraints), not safety. Rewrite the guarantee as "actions invalid under the current simulator mask are prevented", and note that a mask-valid action can still cause starvation, deadlock, or battery depletion. Feasibility filtering is not safety assurance.

**Response:** The remaining draft residue and verified typographical issues have been corrected. The report outline now includes Ethics and Sustainability, the LLM abbreviation is used consistently, run and session terminology is defined in the experimental protocol, and the feasibility guarantee no longer describes mask validation as safety assurance. The Appendix command-line spelling was checked against the experiment runners and is already correct. The previously reported duplicated JSON definition and blank body page are no longer present.

**Action taken:** An explicit generative-AI-use declaration has been added. It identifies OpenAI Codex as support for code development, LaTeX editing, report structuring, and draft review, and OpenAI ChatGPT as occasional support for rephrasing and suggestions concerning report sections. It retains author responsibility for the research, experiments, analysis, interpretation, and final content. The bibliography's imported `number = {null}` field has also been removed.

**Pending action:** Request the assigned UPTEC serial number from the university and replace the `UPTEC XX XXXXX` placeholder before final submission.

---

## Overall

The thesis has a clear engineering core and a sensible comparative intent, and the grounded supervisory loop is worth preserving. The problem is not the work but the way its evidence is currently built and presented: there is no baseline, the runs are unbalanced and reported without uncertainty, the main outcome is not a controlled measure, two sub-RQs compare whole pipelines rather than their named factors, the logged failure evidence is never shown, and the abstracts and conclusion are out of step with the data. As it stands the thesis is not yet defensible, but none of this requires throwing away the framework — most of it is a matched baseline, a balanced re-run (or a careful re-analysis of the logs you already have), and honest reporting.

Please prioritise in this order:

1. **Add matched non-LM baselines and scaffold ablations**, under identical seeds and horizons, so the LM's marginal contribution becomes visible — and remove the unsupported heuristic claim from the Swedish summary unless you can supply the experiment.
2. **Build the run manifest and report uncertainty:** `n` per cell, seeds, attempted/failed/excluded counts, SD/IQR, bootstrap CIs, and seed-level paired plots; relegate the current pooled tables to preliminary status until they can be reconstructed at run level.
3. **Fix the outcome and stopping rule:** re-analyse `deliveries_per_1000_steps` and the time/step metrics you already log, or rerun under a common fixed horizon; report tokens and GPU-seconds, not calls, as cost.
4. **Report the logged failure/grounding metrics** and add invariant tests showing zero out-of-mask execution.
5. **Separate JSON input from structured output** (a 2×2), or explicitly reframe RQ3 as a whole-pipeline comparison; likewise test size within one family or rename RQ1 to "model choice".
6. **Publish an immutable artifact** — code, prompts, exact model digests, configs, seeds, raw data, analysis scripts — and demonstrate one clean-environment reproduction of a headline table.
7. **Rewrite both abstracts and Chapter 6** as explicit, evidence-traceable answers to the main RQ and RQ1–RQ3.

The implemented framework is genuinely worth building on; the revision task is to make its evidence as rigorous and reusable as its engineering. Do the Priority 1–2 work, keep the cautious language you already use in places, and this becomes a solid master thesis.

Best,
Your supervisor
