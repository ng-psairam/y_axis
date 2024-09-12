

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def similarity_scores(resume_keywords_set,job_keywords):
    
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    def get_bert_embedding(text):
    # Tokenize input text and convert to tensor
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Ensure input_ids and attention_mask are passed as kwargs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the embeddings from the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings.numpy()

    def calculate_cosine_similarity_bert(resume_keywords, job_keywords):
        # Combine keywords into sentences
        resume_text = ' '.join(resume_keywords)
        job_text = ' '.join(job_keywords)
        
        # Get BERT embeddings for both resume and job description
        resume_embedding = get_bert_embedding(resume_text)
        job_embedding = get_bert_embedding(job_text)
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity([resume_embedding], [job_embedding])  
        return cosine_sim[0][0]
    
    rows=[]
    # Iterate over job keywords
    for role, keywords in job_keywords.items():
        job_keywords_set = set([i.lower() for i in keywords])

        print(f'\nRole: {role}')
        print('resume_keywords:', resume_keywords_set)
        print('job_keywords:', job_keywords_set)

        print(type(resume_keywords_set),type(job_keywords_set))

        # Calculate Jaccard Similarity
        def jaccard_similarity(set1, set2):
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union

        jaccard_sim = jaccard_similarity(resume_keywords_set, job_keywords_set)
        print('Jaccard Similarity:', jaccard_sim)

        # Calculate Simple Overlap Similarity
        def similarity(set1, set2):
            intersection = len(set1.intersection(set2))
            jd_len = len(set2)
            score = intersection / jd_len
            return score

        overlap_similarity = similarity(resume_keywords_set, job_keywords_set)
        print('overlap_Similarity:', overlap_similarity)

        # Calculate Cosine Similarity using BERT embeddings
        similarity_score_bert = calculate_cosine_similarity_bert(resume_keywords_set, job_keywords_set)
        print(f"Cosine Similarity (BERT Embeddings): {similarity_score_bert:.2f}")

    # Add a new row to the list
        rows.append({
            'resume_keywords': resume_keywords_set,
            'job_keywords': job_keywords_set,
            'overlap_similarity': overlap_similarity,
            'jaccard_similarity': jaccard_sim,
            'cosine_similarity_bert': similarity_score_bert
        })
    df=pd.DataFrame(rows)
    #print(df)
    return df

def main():
    # Sample keywords from a resume
    resume_keywords = {"Data analysis", "Machine Learning", "Python", "SQL", "Data Science"}
    resume_keywords_set = set([i.lower() for i in resume_keywords])

    # Job keywords dictionary
    job_keywords = {
        "Data Scientist": ["Data analysis", "Machine Learning", "Python", "SQL"],
        "Software Engineer": ["Java", "Spring Boot", "Backend development"],
        "Cloud Architect": ["AWS", "CloudFormation", "Cloud architecture"],
        "DevOps Engineer": ["CI/CD", "Docker", "Kubernetes", "Automation"],
        "Cybersecurity Analyst": ["Security protocols", "Penetration Testing", "Risk management"],
        "Machine Learning Engineer": ["TensorFlow", "Python", "ML models"],
        "Frontend Developer": ["HTML", "CSS", "JavaScript", "UI design"],
        "Blockchain Developer": ["Solidity", "Ethereum", "Smart contracts"],
        "Full Stack Developer": ["React", "Node.js", "Full stack development"],
        "AI Research Scientist": ["NLP", "Deep Learning", "AI research"],
        "Product Manager": ["Product lifecycle", "Roadmap planning", "UX/UI"],
        "Associate Product Manager": ["Market analysis", "Competitive research"],
        "Senior Product Manager": ["Agile", "User stories", "Prioritization"],
        "Technical Product Manager": ["API integration", "Technical documentation"],
        "Product Operations Manager": ["Operational efficiency", "Process improvement"],
        "Sales Executive": ["Lead generation", "Client relationship"],
        "Account Manager": ["Customer success", "Sales strategy"],
        "Business Development Manager": ["B2B sales", "Market expansion"],
        "Regional Sales Manager": ["Territory management", "Sales forecasting"],
        "Inside Sales Representative": ["Cold calling", "Sales pitch"],
        "SAP Consultant": ["SAP modules", "ERP implementation"],
        "SAP Project Manager": ["Project management", "SAP implementation"],
        "SAP Analyst": ["SAP FICO", "Data analysis"],
        "SAP Technical Consultant": ["ABAP", "SAP NetWeaver", "Technical documentation"],
        "SAP Solution Architect": ["SAP HANA", "Cloud integration"]      
    }

    df=similarity_scores(resume_keywords_set,job_keywords)
    df.to_excel('similarity_outputs_1.xlsx')


if __name__ == "__main__":
    main()