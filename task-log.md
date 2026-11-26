# Task Log

## Milestone 1

Simone
- helped brainstorm dataset and method ideas
- helped write milestone 1 docs
- created repo

Qlyve
- helped brainstorm dataset and method ideas
- write milestone 1 docs

Carlos
- helped brainstorm dataset and method ideas
- write milestone 1 docs

Nicholas
- helped brainstorm dataset and method ideas
- came up with the PhilSA flood dataset idea and backup
- researched the proposal and feasibility

## Milestone 2

Simone
- wrote the NASA POWER weather fetch and cache (#14)
- built the processed CSV, the storm clusters and the frozen year split (#15)
- checked the issues and PRs against design.md and set the runs.csv format

Qlyve
- implemented Gaussian Naive Bayes from scratch (#17)
- implemented logistic regression with L2 and class weights from scratch (#18)
- wrote the model runner and the first validation runs

Carlos
- wrote the data loader, the split by year and the evaluation metrics (#19)
- wrote the preprocessing and the majority, rain rule and place history baselines (#20)

Nicholas
- wrote the design plan and the data README (#10)
- wrote the shapefile download script, the shapefile reader and the flooded area per tile (#12, #13)
- reviewed and merged #17 to #20, fixed the merge conflicts, wrote the README run steps
- wrote the milestone 2 report

## Milestone 3

Simone
- added the feature-group experiment (#35)
- built the main comparison table and exported the validation scores for the ROC curves (#36)
- applied the Naive Bayes missing-soil fix in her branches and reran the results
- cleaned up the figure script and tests, added requirements.txt, the setup guide and the AI tool disclosure

Qlyve
- added the ROC curve helper and the plotting script, and generated the five validation figures (#39)
- recorded the milestone 3 decisions and the method score analysis in design.md, and added the README results section (#40)

Carlos
- ran the flood threshold sensitivity at 0.5, 1 and 2 percent (#33)
- wrote the per-cluster evaluation and the error analysis (#34)
- moved the shared method settings into src/experiments.py (#38)

Nicholas
- planned and assigned the milestone 3 issues (#22 to #30)
- implemented K nearest neighbors from scratch and recorded the hyperparameters and seeds (#31, #32)
- found the Naive Bayes missing-soil flag issue and fixed it in the shared preprocessing (#37)
- reviewed #33 to #36 and #38, fixed the merge conflicts in #35 and #36, moved #34 to the shared setup, and merged them
- checked main against the milestone 3 requirements and fixed three doc details (#41)

## Milestone 4

Simone
- planned the milestone 4 runner and threshold-tuning work and implemented the pipeline and fair tau selection (#43, #44)
- verified the clean-clone reproduction, the 187-test suite and the gated test-evaluation checks
- reviewed the project-page scaffold (PR #54) and coordinated the milestone 4 deliverables


Qlyve
- helped draft and refine the research content presented on the project page (#47)


Carlos
- built the responsive project page, navigation, branding and MathJax setup (#47)
- integrated research content, validation tables, figures and references

Nicholas
- implemented the clustered bootstrap, the per-cluster analysis and the gated test evaluator (#45, #46)
- reviewed and validated the milestone 4 code pull requests (#50–#53)
