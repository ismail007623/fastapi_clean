from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel,EmailStr,Field
import uuid
from passlib.context import CryptContext
import json
import psycopg2
import os
import shutil
import tempfile


load_dotenv()

db_config={
    "database":os.getenv("DATABASE"),
    "user":os.getenv("USER"),
    "host":os.getenv("HOST"),
    "password":os.getenv("PASSWORD"),
    "port":os.getenv("PORT")

    }

def gen_connection():
    try:
        connection=psycopg2.connect(**db_config)
        print("database conection successfuly")
        return connection
    except Exception as error:
        print(error)

embeddings=GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
)
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)
api_key=os.getenv("PINECONE_API_KEY")
pc=Pinecone(api_key=api_key)

index_name="cricket"

app=FastAPI()
pswd=CryptContext(schemes=["bcrypt"],deprecated="auto")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    email:EmailStr=Field(...,description="Enter your email",examples=["test87@gmail.com"])
    username:str=Field(...,description="Enter username", examples=["ismail_76","waseem78"])
    gender:str=Field(...,description="enter  your gender",examples=["Male or Female"])
    password:str=Field(...,description="Enter your password",examples=["test78#@"])

class Login(BaseModel):
    email:EmailStr=Field(...,description="Enter your email")
    password:str=Field(...,description="enter your password for login")

@app.get('/')
def message():
    return{"message":"welcome to chatbot fastapi"}

@app.post('/signup')
async def sign_up(user:User):
    connection=gen_connection()
    cursor=connection.cursor()
    create='''CREATE TABLE IF NOT EXISTS users(
    user_id UUID PRIMARY KEY not null ,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(255) UNIQUE NOT NULL,
    gender VARCHAR(15),
    password varchar(255) UNIQUE NOT Null)'''



    cursor.execute(create)

    user_id=str(uuid.uuid4())
    hashed_password = pwd_context.hash(user.password)

    insert='''INSERT INTO users(user_id,email,username,gender,password)
    VALUES(%s,%s,%s,%s,%s)'''
    cursor.execute(insert,(user_id,user.email,user.username,user.gender,hashed_password))

    connection.commit()

    cursor.close()
    connection.close()

    return{"message":f"user {user.username} sigin successfully",
           "email":user.email,
           "gender":user.gender,
           }

@app.post('/signin')
async def login(login:Login):
    connection=gen_connection()
    cursor=connection.cursor()
    search='''SELECT user_id , password FROM users WHERE email=%s '''
    cursor.execute(search,(login.email,))
    userid_password=cursor.fetchone()

    if not userid_password:
        raise HTTPException(status_code=401, detail="invalid email or username")
    
    user_id , hash_password =userid_password

    if not pwd_context.verify(login.password, hash_password):
        raise HTTPException(status_code=401, detail="invalid password")


    connection.commit()
    cursor.close()
    connection.close()

    return{
        "message":"user login successfully",
        "user_id":user_id
    }

@app.post('/pdf')
async def pdf_file(
    user_id:str=Form(...,description="enter user_id"),
    file:UploadFile=File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file , tmp)
        tmp_path=tmp.name

    loader=PyMuPDFLoader(tmp_path)

    document=loader.load()
    timestamp=datetime.now()
    namespace=user_id

    for doc in document:
        doc.metadata["user_id"]=user_id
        doc.metadata["namespace"]=namespace
        doc.metadata["filename"]=file.filename
        doc.metadata["uploaded_at"]=timestamp.strftime("%d%m%y %H%M%S")


    vector_store=PineconeVectorStore.from_documents(
        documents=document,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

    connection=gen_connection()

    if not connection:
        raise HTTPException(status_code=401 , detail="postgres not connected successfuly")
    
    try:

        cursor=connection.cursor()
        create='''
            CREATE TABLE IF NOT EXISTS upload(
            id SERIAL PRIMARY KEY NOT NULL,
            user_id UUID NOT NULL UNIQUE,
            files JSONB NOT NULL)'''
        cursor.execute(create)
        connection.commit()

        new_file_record={
            "filename":file.filename,
            "uploaded_at":timestamp.strftime("%d-%m-%y %H%M%S")
        }

        cursor.execute("SELECT files FROM upload WHERE user_id =%s", (user_id,))
        existing_user=cursor.fetchone()

        connection.commit()

        if existing_user:
            current_files=existing_user[0]
            if isinstance(current_files , str):
                current_files=json.loads(current_files)
            if isinstance(current_files, list):
                current_files=[current_files]

            current_files.append(new_file_record)

            cursor.execute(
                "UPDATE upload SET files=%s WHERE user_id=%s",
                 (json.dumps(current_files),user_id)
                        )
            connection.commit()
            return{
                "messgae":f"user {user_id} already esists ",
                "files":current_files

            }
        else:
            new_file = [new_file_record]
            cursor.execute(
            "INSERT INTO upload(user_id,files) VALUES(%s,%s)",
            (user_id,json.dumps(new_file)),)

            connection.commit()

            return{
                "message":f"Data stored succesffuly for user {user_id}",
                "files":new_file
            }
        


    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"database error{e}")
    finally:
        cursor.close()
        connection.close()

class User_chat(BaseModel):
    user_id:str=Field(..., description="enter your user id")
    bot_id:str | None=None
    query:str=Field(..., description="enetr your question")

@app.post("/chat")
async def chat(user:User_chat):
     connection=gen_connection()

     if not connection:
        raise HTTPException(status_code=402, detail="Database not connected")
    
     cursor=connection.cursor()

     cursor.execute("SELECT user_id  FROM users WHERE user_id=%s ",(user.user_id,))
     id=cursor.fetchone()

     connection.commit()
 
     if not id:
        cursor.close()
        connection.close()
        raise HTTPException(status_code=404 , detail="user id not exits please login")
    
     create='''CREATE TABLE IF NOT EXISTS bot(
     id SERIAL PRIMARY KEY NOT NULL,
     user_id Varchar NOT NULL ,
     bot_id varchar  NOT NULL,
     user_message TEXT NOT NULL,
     ai_message TEXT NOT NULL,
     created_at TIMESTAMP DEFAULT NOW())'''
    
     cursor.execute(create)
     connection.commit()

     if not user.bot_id or user.bot_id.strip().lower() == "string":

      bot_id=f"bot_{uuid.uuid4().hex}"

     else:

        bot_id = user.bot_id
        cursor.execute("SELECT 1 FROM bot WHERE user_id=%s AND bot_id=%s LIMIT 1", (user.user_id, bot_id))
        session = cursor.fetchone()
        if not session:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=404, detail="Invalid bot_id for this user")

     history='''
    SELECT user_message , ai_message
    FROM bot WHERE bot_id=%s 
    ORDER BY created_at DESC
    LIMIT 10'''
    
     cursor.execute(history,(bot_id,))
     history_user_ai_msg=cursor.fetchall()

     connection.commit()

     history_data=""

     for user_msg ,ai_msg in reversed(history_user_ai_msg):
        history_data +=f"user:{user_msg}\n Ai:{ai_msg}\n"

     try:

        vector_store=PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=user.user_id
        )

        docs=vector_store.similarity_search(user.query, k=3)
     except Exception as error:
        raise HTTPException(status_code=401 , detail=f"pinecone retriver error:{str(error)}")
    
     pdf_data="\n".join([doc.page_content for doc in docs])

     prompt = f"""
    You are a helpful Assistant.

    conversation so far:
    {history_data}

    context from documents:
    {pdf_data}

    question: {user.query}
    """
     try:
        ai_response_obj =await llm.ainvoke(prompt)
        ai_response= ai_response_obj.content if hasattr(ai_response_obj, "content") else str(ai_response_obj)

     except Exception as error:
        raise HTTPException(status_code=401, detail=f"LLM error: {str(error)}")

    


     insert='''
    INSERT INTO bot (user_id,bot_id,user_message,ai_message,created_at)
    VALUES(%s,%s,%s,%s,%s)'''

     cursor.execute(insert,(user.user_id,bot_id,user.query,ai_response,datetime.now()))

     connection.commit()

     cursor.close()
     connection.close()


     return{
        "Question":user.query,
        "AI_message":ai_response,
        "bot_id":bot_id
    }