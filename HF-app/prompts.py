sentiments_prompt = """
Based on the following product reviews and their sentiment scores (0-100), 
identify what people consistently like and dislike about the product. It can be about the product, the packaging, the delivery, the customer service, etc 
but it doesn't have to be restricted to these. These are just examples on what people can like or dislike about the product.

So your task is to aggregate all the reviews together and identify 3 likes and 3 dislikes at most
in the order of what people like and dislike the most. If some aspects of the product are liked/ disliked the most in the reviews,
then they should be included first in the list and so on. 

Remember your job is to aggregate the reviews and provide general likes and dislikes about the product. 
Don't be specific to the individual reviews.

For example, likes can be (but not limited to):
- The product is easy to use
- The product is durable
- customer service is helpful.
- The shipping was fast.

Dislikes can be (but not limited to):
- The product was not secure
- The customer service helpline was slow to respond

These are just examples. You can include more likes and dislikes as long as they are relevant to the product. 
While creating your output remember these points:
- Remember that products with higher sentiment scores are more likely to be liked by the customers and lower scores are more likely to be disliked. 
- If 2 or more likes/dislikes are similar, then you can combine them into one. Make sure your each of your likes/dislikes are unique and not similar to each other. 
- Make sure your output is in the order of importance.
- Dont make up likes/dislikes. Your likes/dislikes should be based on the reviews of this product ONLY. You can find those reviews in between the 
<Product Reviews> and </Product Reviews> tags.

<Product Reviews>
{review_context}
</Product Reviews>
"""