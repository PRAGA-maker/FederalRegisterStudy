# Pie graph, what % of total comments in 2024 are from: [Undecided/Anonomyous], [Ordinary Citizen], [Organization/Corporation], [Academic, Industry, or otherwise Expert (including small and local business)], [Political Consultant or Lobbyist], 
# Make a 2nd plot: By agency as well, make a XY plot (similar to the 4 quadrants plot used in market analysis) writing out the agencies as a XY point where the X is [Ordinary Citizen, Organiztion/Corporation] and Y is [Academic, Expert to Political Consultant or Lobbyist]
# We can compute these using the openAI api key: openai key: [OPENAI_API_KEY]  and using the gpt-5-nano model with a small, lightweight prompt+output with the prompt standardizing the output. 

#dont hardcode api key, get it from os.environ.get("OPENAI_API_KEY")


#X,y plot of how many comments total vs. how many of the comments are from (multiple lines) Organization/Corporations, Ordinary Citizen, Academic or Expert, Undecided expected to show how less comments >> less ordinary citizen input

#practically, because we don't want to rerun the openai LLM calls, we need to create a database .csv file called "makeup_data.csv" that contains:
# - document_number
# - makeup type (Undecided/Anonomyous, Ordinary Citizen, Large Organization/Corporation, Academic, Industry, or otherwise Expert (including small and local business), Political Consultant or Lobbyist)

#we need to write one script that runs the agent loop across the /output/XXXX.csv containing by-year data, and for the year 2024 file it runs through it all, writes that CSV

#then, there heeds to be a seperate script which looks at the makeup_data.csv and creates the plots, and for the XY plot including agencies if it isn't in the dataset then running the neccessary api calls to get the agency name. 