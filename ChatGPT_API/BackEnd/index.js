const apiKey = "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"

// const { Configuration, OpenAIApi } = require("openai");

// const configuration = new Configuration({
//   apiKey: "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",
// });
// const openai = new OpenAIApi(configuration);

// async function apiCall() {
//   const completion = await openai.createChatCompletion({
//       model: "gpt-3.5-turbo",
//       messages: [{role: "user", content: "Hello world"}],
//       // model: "text-davinci-003",
//       // prompt: "Hello world",
//     });
//   console.log(completion.data.choices[0].message['content']);
//   // console.log(completion);
// }

// apiCall();

const { Configuration, OpenAIApi } = require("openai");

const configuration = new Configuration({
  apiKey: apiKey,
});
const openai = new OpenAIApi(configuration);

async function apiCall() {
  const completion = await openai.createChatCompletion({
    model: "gpt-3.5-turbo",
    messages: [{"role": "system", "content": "You are a helpful assistant."}, {role: "user", content: "Hello world"}],
  });
  console.log(completion.data.choices[0].message);
}
apiCall();
