# Total Variation based image Resurfacing (TVR)

Adversarial patches threaten visual AI models in the real world. The number of patches in a patch attack is variable and determines the attack’s potency in a specific environment. Most existing defenses assume a single patch in the scene, and the multiple-patch scenario is shown to overcome them. This paper presents a model-agnostic defense against patch attacks based on total variation for image resurfacing (TVR). The TVR is an image-cleansing method that processes images to remove probable adversarial regions. TVR can be utilized solely or augmented with a defended model, providing multi-level security for robust prediction. TVR nullifies the influence of patches in a single image scan with no prior assumption on the number of patches in the scene. We validate TVR on the ImageNet-Patch benchmark dataset and with real-world physical objects, demonstrating its ability to mitigate patch attacks.

<img src="Figures/TVD.pdf"/> 
