---
license: bsd-3-clause
language:
- en
---

<a href="https://github.com/VLKEB/VLKEB">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/65d86142a3c18e931641be25/gUarL9URv_QFlXN84L4xG.jpeg" alt="Logo" width="600">
</a>


<h2>VLKEB: A Large Vision-Language Model Knowledge Editing Benchmark</h2>

arxiv: [https://arxiv.org/abs/2403.07350](https://arxiv.org/abs/2403.07350)

github: [https://github.com/VLKEB/VLKEB](https://github.com/VLKEB/VLKEB)

Recently, knowledge editing on large language models (LLMs) has received considerable attention. Compared to this, editing Large Vision-Language Models (LVLMs) faces extra challenges from diverse data modalities and complicated model components, and data for LVLMs editing are limited. The existing LVLM editing benchmark, which comprises three metrics (Reliability, Locality, and Generality), falls short in the quality of synthesized evaluation images and cannot assess whether models apply edited knowledge in relevant content. Therefore, we employ more reliable data collection methods to construct a new Large **V**ision-**L**anguage Model **K**nowledge **E**diting **B**enchmark, **VLKEB**, and extend the Portability metric for more comprehensive evaluation. Leveraging a multi-modal knowledge graph, our image data are bound with knowledge entities. This can be further used to extract entity-related knowledge, which constitutes the base of editing data. We conduct experiments of different editing methods on five LVLMs, and thoroughly analyze how do they impact the models. The results reveal strengths and deficiencies of these methods and hopefully provide insights for future research.
