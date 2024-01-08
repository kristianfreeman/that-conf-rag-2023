import {
	ChatCloudflareWorkersAI,
	CloudflareVectorizeStore,
	CloudflareWorkersAIEmbeddings,
} from "@langchain/cloudflare";

import { RetrievalQAChain } from "langchain/chains"

import { Hono } from "hono"

const app = new Hono()

app.get("/", async c => {
	const query = c.req.query("query") || "Hello World"

	const embeddings = new CloudflareWorkersAIEmbeddings({
		binding: c.env.AI,
		modelName: "@cf/baai/bge-small-en-v1.5"
	})

	const store = new CloudflareVectorizeStore(embeddings, {
		index: c.env.VECTORIZE_INDEX
	})

	const storeRetriever = store.asRetriever()

	const model = new ChatCloudflareWorkersAI({
		cloudflareAccountId: c.env.CLOUDFLARE_ACCOUNT_ID,
		cloudflareApiToken: c.env.CLOUDFLARE_API_TOKEN,
		model: "@cf/mistral/mistral-7b-instruct-v0.1"
	});

	const chain = RetrievalQAChain.fromLLM(model, storeRetriever)

	const res = await chain.invoke({ query })

	return c.json(res)
})

app.post("/add", async c => {
	const embeddings = new CloudflareWorkersAIEmbeddings({
		binding: c.env.AI,
		modelName: "@cf/baai/bge-small-en-v1.5"
	})

	const store = new CloudflareVectorizeStore(embeddings, {
		index: c.env.VECTORIZE_INDEX
	})

	const { id, text } = await c.req.json()

	await store.addDocuments([{ pageContent: text }], { ids: [id] })

	return c.text("Created", { status: 201 })
})

app.onError((err, c) => {
	return c.text(err)
})

export default app
