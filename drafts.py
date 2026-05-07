 # print(f"Uploading {len(documents)} chunks to Pinecone. This may take a moment depending on the batch size...")
    # # Generate a unique ID for every single chunk
    # unique_ids = [str(uuid.uuid4()) for _ in documents]

    # # This single line handles the embedding generation AND the batched upload to Pinecone

    #     # PineconeVectorStore.from_documents(
    # #     documents=documents,
    # #     embedding=embeddings_model,
    # #     index_name=index_name,
    # #     ids=unique_ids
    # # )
    # logging.basicConfig(level=logging.DEBUG)
    # try:
    #     vectorstore = PineconeVectorStore(
    #         index_name=index_name,
    #         embedding=embeddings_model
    #     )

    #     vectorstore.add_documents(documents, ids=unique_ids)
    #     logging.basicConfig(level=logging.DEBUG)
    # except Exception as e:
    #     print(f"Error: {e}")
    # print("Upload complete! Your vector database is ready for retrieval.")