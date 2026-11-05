# Task Log

## Milestone 1

**What was done.** The group selected the PhilSA flood extent dataset as labels with NASA POWER weather as features and framed a tile-level flood classification task. The dataset inventory and weather endpoints were verified, the instructor confirmed the two sources count as one task, and the group chose three method families with an evaluation protocol. The submission repository was created and the instructor was added as a collaborator.

**Who did it.** Nicholas O. Pacoma wrote the proposal. Simone Pauline M. Dagondon verified dataset and weather access, reviewed the proposal, and wrote the methods, evaluation, and repository plans. Andrew Qlyve Paul C. Bundoc and Juan Carlos J. Bernardino completed their assigned setup tasks.

**Problems encountered.** The dataset claim is first come, first served and must be confirmed with the instructor. Same-day weather would leak information about rain after the satellite pass, which the pre-pass hourly design resolves.

**Plan for the next milestone.** Build the shapefile reader and frozen tile grid, fetch the weather features, freeze the processed table and split file, implement Gaussian Naive Bayes and K nearest neighbors from scratch, and implement the metrics from scratch.
