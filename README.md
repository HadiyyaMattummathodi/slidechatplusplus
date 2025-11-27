# SlideChat++

SlideChat++ is an enhanced multimodal large language model for Whole Slide Image (WSI) understanding.  
This project extends the original **SlideChat** framework by integrating the **FOCUS** adaptive visual token compression approach to remove redundant patch tokens and improve training/inference efficiency.

## 🔬 Key Features
- **FOCUS-based token redundancy removal** to significantly reduce visual token count.
- **Fine-tuning on 5000 WSIs** from the SlideChat dataset.
- **Evaluation on SlideChat and WSI-LLaVA benchmarks** for histopathology image classification and question answering.
- **Efficient multimodal encoder–decoder architecture** for large WSI pathology tasks.

## 📘 Included in this Repository
- Training and inference pipelines for SlideChat++
- Token compression visualization scripts (FOCUS adaptation)
- Preprocessing and dataset loaders for SlideChat & WSI-LLaVA
- Evaluation and classification scripts for unseen WSIs
- Code for generating final report figures and metrics

## 📄 Related Papers
- SlideChat (CVPR 2025)
- WSI-LLaVA (ICCV 2025)
- FOCUS (CVPR 2025)

---

This repository is part of a course project on developing improved LLM-based histopathology image analysis models.
