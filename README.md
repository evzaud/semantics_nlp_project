# AMR-based Answer Generation
By Caitlyn Chen, Justin Chen, Tiffeny Chen, Yibing Chen, and Evan Zauderer

## Goal

Standard manual question answer (QA) generation is labor and time-intensive. We propose two automated QA generation frameworks, which leverage abstract meaning representations (AMRs) in order  to provide semantic information to our models. Our first proposal revolves around the creation of templates that map certain “question phrases” to the likeliest AMR roles for the answer. Our second proposal utilizes a Sequence to Sequence (seq2seq) model in order to learn answers from the input AMR representations.

The goal of this project is to automatically generate valid, logical, and fluent answers. Usage scenarios include automatic tutoring systems and chatbots.
## Resources
### Dataset: QAMR

For this project we use the QAMR dataset, which contains about 5,000 annotated sentences (from Newswire and Wikipedia) and 100,000 question-answer pairs on those sentences created by crowdsourcing workers.

### AMRLib

For all translations between English and AMR we use AMRLib, a Python library that intends to make AMR parsing, generation and visualization simple.

## Methods

### Method 1: AMR+templates

Our first method utilizes a rule-based lookup table. Essentially, we scan a question for certain common question phrases. Then, our lookup table tells us the likeliest AMR role(s) that would contain the answer if we were to search the context. We then search the AMR graph of the context for those role(s) and parse out the AMR associated with that role. We finally convert that section from AMR to English and use it as our proposed answer.

### Method 2: AMR+seq2seq

Our second method is a more classic "intelligent systems" approach, using a seq2seq neural network. We concatenate the tokenized AMR graphs for the question and context, and then pass that to our neural network. We trained the neural network to learn a tokenized version of the answer to the given question.

## Results
Our results (found in results.txt) shed a lot of light on AMR and our 2 methods. Both methods yield rather low values for the Exact Match Check, with Method 1 yielding an exact match to the gold standard 12.2% of the time, compared to Method 2's 13.6%. These low values make sense, since converting an English sentence into AMR will cause a loss of some portion of the English sentence, notably the tense. Therefore, it is unsurprising that both methods often did not yield answers that perfectly matched the gold standard.
However, the great benefit to using Method 2 became clear when looking at the BLEU scores the 2 methods yielded. Method 1 yielded a BLEU score of 0.189, while Method 2 yielded a BLEU score of 0.373. This large discrepancy shows that Method 2 is learning the general answer far better than Method 1, with Method 1 seemingly often learning a completely incorrect answer (as the BLEU score of 0.189 is only marginally better than getting the answer completely right with probability 0.122). However, there is a notable jump in the BLEU score for Method 2, showing that even when the answer provided by Method 2 does not exactly match the gold standard, it does yield higher precision (BLEU is a precision-based metric, not recall-based).
These metrics were chosen as they seemed to be the standard, at the very least, for the similar work seen in our research process.

## Future Work
- Using METEOR evaluation, which uses precision and recall (compared to solely precision, like BLEU). It is associated with a higher correlation to human judgement 
- Jointly performing question generation and QA
- Test our approaches on other datasets, perhaps with more complex sentences and/or domain-specific data 
- Combined pipeline of generating answers using templates and seq2seq approaches
- Add additional reference sentences for valid answers. For example, we could change the gold standard answer to an AMR graph, and then back into English, using the result as an additional reference sentence.
