# Week 06 — Site & Unit Segmentation, Risk Drivers & Fari Extension

## Work Completed

This week, I developed an exploratory robot-segmentation and site-risk analysis using the corrected fleet dataset. I aggregated mission, telemetry, fault, and maintenance records into a robot-level feature table covering 50 units. To avoid defining clusters directly by their outcomes, I excluded fault counts, maintenance counts, and fault rates from the clustering features. I standardized eight operational and condition features and used principal component analysis to reduce redundancy. The first four components explained approximately 81.4% of the variance.

I evaluated K-Means solutions using silhouette scores, the Davies–Bouldin index, inertia, and stability. Although K=5 achieved a stronger silhouette score, it created two very small groups. I selected K=3 for more balanced and interpretable operational segments of 10, 20, and 20 robots. Random-initialization stability was strong, but 80% subsampling produced a mean adjusted Rand index of 0.768, indicating sensitivity to changes in the robot sample.

I compared historical fault frequencies across clusters and sites. Cluster 0 showed the highest wheel wear and historical fault frequency despite relatively healthy battery capacity. This demonstrated why battery health alone is insufficient to summarize fleet condition. I also fitted a Poisson regression with operating-hours exposure and operational covariates. SITE_07 had an elevated estimated fault-rate ratio relative to SITE_01, but SITE_07 was also the only rough-terrain site, making site and terrain effects inseparable.

## Reflection: Did Segmentation Add Value Beyond the Failure Model?

Segmentation provided a different level of insight from the existing failure-prediction model. Weeks 3 and 4 emphasized pre-fault signals and predictive feature importance, whereas Week 6 identified operationally distinct groups and differences in historical site-level fault frequency. For example, the cluster with the lowest battery capacity did not have the highest overall fault rate. This finding suggests that maintenance planning should consider several condition dimensions rather than a single health indicator.

The segmentation is not a substitute for the failure model. Its sample sensitivity and retrospective design limit its use for fixed risk labels. Moreover, the earlier signal and model analyses used a different data vintage from the corrected Week 6 dataset, so cross-week conclusions remain qualitative until consistently rerun.

## Fari Extension and Next Steps

I assessed which workflow components could conceptually transfer to Fari, including data validation, feature standardization, PCA, clustering, and fault-rate analysis. No Fari-specific dataset was available in the current project directories, so I documented schema requirements and a validation plan rather than claiming completed transfer testing. Next steps include obtaining Fari data, verifying feature definitions, repeating stability tests, and measuring whether segmentation improves maintenance decisions beyond the existing predictive model.
