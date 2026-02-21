<a id="readme-top"></a>
<br />
<div align="center">


  <a href="https://github.com/hireshBrem/browserbrain-ai">
    <img src="https://img.shields.io/github/last-commit/hireshBrem/browserbrain-ai?style=flat-square" alt="Last commit">
  </a>
  <img src="https://img.shields.io/badge/Node.js-18+-green?style=flat-square&logo=node.js" alt="Node.js">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.121+-green?style=flat-square&logo=fastapi" alt="FastAPI">
    <!-- PROJECT LOGO -->

<h1 align="center">
    <br>
    <img src="./client/public/logo.png" alt="Browsor Agent Logo" width="100">
    <h3 align="center">{Name}</h3>
</h1>


  <p align="center">
    {subtitle}

<a href="">View Demo</a>

![Resac Agent Screenshot](./client/public/ss.png)
    <br />
    <br />

  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

{about}

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Before running Resac Agent, ensure you have the following installed:

- **Node.js 18+**
- **Python 3.12+**
- **Docker & Docker Compose** (optional, for containerized setup)
- **uv** (Python package manager) 

You'll also need API keys for:
- **OpenAI API Key** 
- **Perplexity API Key** (optional, for research agent) 

### Installation

#### Option 1: Using Docker Compose (Recommended)

The simplest way to get everything running at once.

1. Clone the repository
   ```sh
   git clone https://github.com/hireshBrem/resac-agent.git
   cd resac-agent
   ```

2. Create a `.env` file in the root directory with your API keys:
   ```sh
   OPENAI_API_KEY=sk-your-openai-key-here
   PERPLEXITY_API_KEY=your-perplexity-key-here
   ```

3. Start the services using Docker Compose:
   ```sh
   docker compose watch
   ```

4. Open your browser and navigate to:
   - **Frontend**: `http://localhost:3000`
   - **Backend API**: `http://localhost:4000`

5. See the logs/stop the containers:
    ```sh
    docker compose logs -f
    docker compose down
    ```

That's it! Both the client and server will be running in containers.

#### Option 2: Manual Setup (Client & Server Separately)

For development or if you prefer running services individually.

##### Backend Server Setup

1. Navigate to the server directory:
   ```sh
   cd server
   ```

2. Create a virtual environment and install dependencies:
   ```sh
   uv venv
   uv sync --frozen
   ```

3. Set your environment variables:
   ```sh
   export OPENAI_API_KEY=sk-your-openai-key-here
   export PERPLEXITY_API_KEY=your-perplexity-key-here
   ```

4. Start the FastAPI server:
   ```sh
   uv run uvicorn main:app --host 0.0.0.0 --port 4000 --reload
   ```

   The server will be available at `http://localhost:4000`

##### Frontend Client Setup

1. In a new terminal, navigate to the client directory:
   ```sh
   cd client
   ```

2. Install dependencies:
   ```sh
   npm install
   ```

3. Start the development server:
   ```sh
   npm run dev
   ```

   The client will be available at `http://localhost:3000`

4. Open `http://localhost:3000` in your browser and start using Resac Agent!



<!-- USAGE EXAMPLES -->
## Usage

The Resac Agent processes research and action tasks through a multi-agent supervisor that streams state updates as agents work through different tasks. Here's an example of what the streaming output looks like:

### Example: Research & Excel Generation Stream

```json
[
    {
        "id": "1763586936951",
        "role": "user",
        "content": "Research the latest AI trends"
    },
    {
        "id": "17635869385550.22294938570853284",
        "role": "assistant",
        "content": "Research",
        "type": "routing",
        "agent": "supervisor",
        "data": {
            "decision": "Research",
            "reasoning": "Routing to specialized agent"
        },
        "timestamp": "2025-11-19T21:15:38.553430Z"
    },
    {
        "id": "17635869385790.31236956623979384",
        "role": "assistant",
        "content": "",
        "type": "agent_start",
        "agent": "Research",
        "data": {
            "task": "Research the latest AI trends"
        },
        "timestamp": "2025-11-19T21:15:38.554605Z"
    },
    {
        "id": "17635869582720.034585653918641324",
        "role": "assistant",
        "content": "Research agent completed processing",
        "type": "agent_end",
        "agent": "Research",
        "data": {
            "status": "success",
            "summary": "Research agent completed processing",
            "response": {
                "messages": [
                    "content='Research the latest AI trends' additional_kwargs={} response_metadata={} id='3a0ad7b6-998b-47df-9147-698a8a2944e7'",
                    "content='The latest AI trends in 2025 center on greater **AI autonomy, multimodality, agent-based systems, enterprise customization, and efficiency**. These trends are transforming both consumer and enterprise landscapes, driving innovation across industries and reshaping the role of AI in daily life and business.\n\nKey trends include:\n\n- **Agentic AI**: AI agents are evolving from simple chatbots to autonomous systems capable of planning and executing multi-step workflows, acting as \"virtual coworkers\" that can handle complex tasks with minimal human intervention[1][4][6]. This agentic approach enables businesses to automate more processes, increasing productivity and allowing human workers to focus on higher-value activities.\n\n- **Multimodal AI**: New models can process and integrate multiple types of data—including text, images, video, and audio—enabling richer, more human-like understanding and decision-making[3][14]. For example, devices can now identify people and objects in photos while understanding context and intent, making them far more versatile.\n\n- **Enterprise Customization and Security**: Organizations are increasingly adopting proprietary AI models tailored to their specific data and needs, enhancing competitive advantage and ensuring data privacy[2][9]. Enterprises also demand robust performance, profitability, and security, driving innovation in AI infrastructure, custom silicon, and cloud migration[2].\n\n- **Advances in Reasoning and Specialized Models**: Large language models (LLMs) and \"frontier models\" are being optimized for better reasoning, efficiency, and performance, often with smaller architectures that rival much larger models thanks to improved data curation and synthetic training[1][5][10]. Mixture-of-experts models and world models are gaining traction for their ability to handle nuanced, context-rich tasks[10].\n\n- **Scientific and Societal Impact**: AI is accelerating breakthroughs in healthcare, materials science, and sustainability. Examples include AI-driven drug discovery and protein simulation, which speed up research and open new possibilities for solving complex global challenges[1].\n\n- **Generative AI Integration**: Generative AI continues to see widespread adoption, with organizations allocating larger portions of their IT budgets to integrate genAI into workflows, products, and services[3]. This trend is expected to generate significant economic value and transform creative and analytical tasks.\n\n- **Physical and Sovereign AI**: The rise of physical AI (AI integrated with robotics or IoT) and sovereign AI (locally governed and controlled AI infrastructures) is also notable, as organizations and governments seek more control over their data and AI operations[6].\n\n- **Accessibility and Affordability**: Open-weight and open-source models are narrowing the performance gap with proprietary solutions, making advanced AI more accessible and affordable to a wider range of users and organizations[5].\n\n- **Evaluation and Measurement**: As adoption grows, there\\'s a greater focus on tools and systems to measure AI efficacy, safety, and impact, ensuring responsible use and continuous improvement[2].\n\nIn summary, 2025 marks a shift from experimental adoption to meaningful integration of AI—especially agentic and multimodal systems—driven by advances in reasoning, customization, efficiency, and a focus on real-world impact[1][2][3][4][5][6][9][10][14].' additional_kwargs={} response_metadata={'finish_reason': 'stop', 'model_name': 'sonar-pro', 'model_provider': 'openai'} id='lc_run--e6c8e2ba-d127-4243-9544-55abbca10101' usage_metadata={'input_tokens': 75274, 'output_tokens': 203524, 'total_tokens': 278798, 'input_token_details': {}, 'output_token_details': {}}"
                ]
            }
        },
        "timestamp": "2025-11-19T21:15:58.270925Z"
    },
    {
        "id": "17635869594840.5104706015044463",
        "role": "assistant",
        "content": "FINISH",
        "type": "routing",
        "agent": "supervisor",
        "data": {
            "decision": "FINISH",
            "reasoning": "Task completed"
        },
        "timestamp": "2025-11-19T21:15:59.483022Z"
    }
]
```

Each agent node (Research, Excel, PowerPoint, etc.) streams its progress in real-time, allowing the client to display partial results as they're generated. Use the `/supervisor/stream` endpoint to get these updates.

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AGIFeature`)
3. Commit your Changes (`git commit -m 'Add some AGIFeature'`)
4. Push to the Branch (`git push origin feature/AGIFeature`)
5. Open a Pull Request

### Contributors:

<a href="https://github.com/hireshBrem/browserbrain-ai/graphs/contributors">
  <img width="36" height="36" src="https://contrib.rocks/image?repo=hireshBrem/browserbrain-ai" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.


<!-- CONTACT -->
## Contact

For questions, feedback, or support, feel free to reach out:

- **X (Twitter)**: [@hiresh_b](https://x.com/hiresh_b)
- **GitHub**: [hireshBrem](https://github.com/hireshBrem)
- **Email**: Open an issue on this repository


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project is built on top of these amazing technologies:

* [Next.js](https://nextjs.org/) - Frontend framework
* [FastAPI](https://fastapi.tiangolo.com/) - Backend web framework
* [LangGraph](https://langchain-ai.github.io/langgraph/) - Multi-agent orchestration
* [OpenAI](https://openai.com/) - Language models
* [Anthropic](https://www.anthropic.com/) - Claude API
* [Browser-Use](https://github.com/browser-use/browser-use) - Browser automation
