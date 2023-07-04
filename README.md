# Total Variation based image Resurfacing (TVR)

By [Abhijith Sharma](https://www.linkedin.com/in/abhijith-sharma/), [Phil Munz](https://www.linkedin.com/in/philmunz/), [Apurva Narayan](https://scholar.google.com/citations?user=e5OCZ1cAAAAJ&hl=en&authuser=2)

Code for "[Assist Is Just as Important as the Goal: Image Resurfacing to Aid Model’s Robust Prediction]()" submitted in WACV 2024. 

<img src="./Figures/TVD.PNG"/> 

**Takeaways**: 
1. Adversarial patches threaten visual AI models in the real world.
2. Number of patches in an attack is variable and determines its potency in a specific environment.
3. Existing defenses assume a single patch in the scene, and the multi-patch scenario is shown to overcome them.
4. The TVR is a model-agnostic defense against single and multi-patch attacks.
5. The TVR is an image-cleansing method that processes images to remove probable adversarial regions.
6. Nullifies the influence of patches in a single image scan with no prior assumption on the number of patches. 


