`price` in the `transactions` table appears to be at an incorrect scale. 
 - A Kaggle user suggested multiplying values in this column by 590 to get correct prices
     - https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/310496
     - SQL query used to update `price` column:
         - UPDATE transactions SET price = price * 590;