# Email Classification Q&A

## If you only had 200 labeled replies, how would you improve the model without collecting thousands more?

I'd use data augmentation techniques to artificially expand the dataset. Transfer learning would be crucial - start with a pre-trained model and fine-tune it on our small dataset. 

## How would you ensure your reply classifier doesn't produce biased or unsafe outputs in production?

I would implement confidence thresholds where low-confidence predictions get human review/re-classification instead of automatic classification. We could also maintain a feedback loop where users can report misclassifications. One more implementation could be adding guradrails/llm-as-a-judge to the model, so every classification would undergo extra check to see if it isnt biased or has unsafe outputs.

## Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?

I'd use few-shot prompting with 3-5 examples of great personalized openers that reference specific company details, recent news, etc. Structure the prompt to require recipient specific inputs like company size, industry, recent achievements, and the recipient's role, then explicitly instruct the model to reference these details. Include constraints like "avoid generic phrases like 'I hope this email finds you well'" and "mention something specific about their company that shows research was done." To avoid hallucination, i can force it to only reference provided fields only.
