/* frontend.h -- a model's request-facing half: tokenizer, chat template (with its reply parser),
 * grammar compiler and the request limits.
 *
 * It has a lifetime of its own, independent of the model runtime: stopping the runtime
 * (Engine::stop_execution) never touches it. A request takes a FrontendLease when it is
 * generated and drops it only after its response is over, final stream chunks and reap included,
 * so the incremental decoder and reply stream it holds stay valid however the runtime fares. The
 * frontend is freed with the last lease.
 *
 * Nothing in it may point into the Engine: a model is deactivated (ModelRouter) and its Engine
 * destroyed while a response's lease can still be alive. The bridges own what they read (the
 * tokenizer and the grammar compiler share the vocabulary, the template its parsed source), and
 * the vocabulary is a copy, not a view into the mapped container. */
#pragma once
#include "iface.h"
#include "oai.h"

#include <memory>

namespace rad {
namespace server {

struct ModelFrontend {
    std::shared_ptr<ITokenizer>       tok;       /* required to serve text */
    std::shared_ptr<IChatTemplate>    chat;      /* absent -> /v1/chat/completions is 501 */
    std::shared_ptr<IGrammarCompiler> grammar;   /* absent -> response_format is 501 */
    /* Filled by the Server from its ServerOptions. */
    OaiLimits                         limits;
};

using FrontendLease = std::shared_ptr<const ModelFrontend>;

}  /* namespace server */
}  /* namespace rad */
