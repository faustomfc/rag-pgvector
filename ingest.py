import click
from src.models.embeddings import load_model, generate_embeddings
from src.persistence.chunks import load_pdfs, index_chunks, create_hnsw_index
from src.persistence.engine import build_engine, initialize_database
from src.utils.constants import EMBEDDING_MODEL, BATCH_SIZE

@click.command()
@click.option("--db-host", type=click.STRING, envvar="DB_HOST", required=True)
@click.option("--db-port", type=click.STRING, envvar="DB_PORT", required=True)
@click.option("--db-name", type=click.STRING, envvar="DB_NAME", required=True)
@click.option("--db-user", type=click.STRING, envvar="DB_USER", required=True)
@click.option("--db-password", type=click.STRING, envvar="DB_PASSWORD", required=True)
@click.option("--pdf-folder-location", type=click.STRING, envvar="PDF_FOLDER_LOCATION", required=True)


def main(pdf_folder_location, db_host, db_port, db_name, db_user, db_password):

    # 1. Database
    engine = build_engine(db_user=db_user, db_password=db_password, db_host=db_host, db_port=db_port, db_name=db_name)
    initialize_database(engine)

    # 2. PDFs → chunks
    chunks = load_pdfs(folder=pdf_folder_location, engine=engine)
    
    # 3. Embeddings
    model = load_model(embedding_model=EMBEDDING_MODEL)
    textos = [c['content'] for c in chunks]
    embeddings = generate_embeddings(model, textos)

    # 4. Index
    index_chunks(engine=engine, chunks=chunks, embeddings=embeddings, batch_size=BATCH_SIZE)

    # 5. HNSW index
    create_hnsw_index(engine)


if __name__ == "__main__":
    main()
