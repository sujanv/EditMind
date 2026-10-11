// EditMind Dashboard Client Logic

document.addEventListener("DOMContentLoaded", () => {
  // Tab switching
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });

  // Preset Chips
  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
      document.getElementById("edit-prompt").value = chip.dataset.prompt;
      document.getElementById("subject-input").value = chip.dataset.subject;
      document.getElementById("target-input").value = chip.dataset.target;
      document.getElementById("ground-truth-input").value = chip.dataset.gt;
    });
  });

  // Trace Button
  document.getElementById("btn-trace").addEventListener("click", async () => {
    const prompt = document.getElementById("edit-prompt").value;
    const subject = document.getElementById("subject-input").value;
    const target = document.getElementById("ground-truth-input").value;

    const summaryEl = document.getElementById("trace-summary");
    const container = document.getElementById("heatmap-container");

    summaryEl.innerHTML = `<p style="color:#89b4fa;">Computing causal mediation trace across all layers and token representations...</p>`;
    container.innerHTML = "";

    try {
      const res = await fetch("/api/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, subject, target, model_name: "toy_causal_lm" })
      });
      const data = await res.json();

      summaryEl.innerHTML = `
        <div style="background:#181825; padding:12px; border-radius:8px; border:1px solid #45475a; margin-bottom:12px;">
          <strong>Peak Causal Site:</strong> Layer ${data.critical_layer}, Token <code>"${data.critical_token}"</code><br>
          <strong>Max Indirect Effect (AIE):</strong> ${data.max_indirect_effect}<br>
          <span style="font-size:12px; color:#a6adc8;">This identifies the feedforward MLP layer where factual associations are stored.</span>
        </div>
      `;
      container.innerHTML = data.svg_heatmap;
      document.querySelector('[data-tab="tab-trace"]').click();
    } catch (err) {
      summaryEl.innerHTML = `<p style="color:#f38ba8;">Error running trace: ${err.message}</p>`;
    }
  });

  // Edit Form Submit
  document.getElementById("edit-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const prompt = document.getElementById("edit-prompt").value;
    const subject = document.getElementById("subject-input").value;
    const target_new = document.getElementById("target-input").value;
    const ground_truth = document.getElementById("ground-truth-input").value;
    const method = document.getElementById("method-select").value;

    const probContainer = document.getElementById("prob-container");
    probContainer.innerHTML = `<p style="color:#89b4fa;">Executing ${method.toUpperCase()} knowledge edit on model memory...</p>`;

    try {
      const res = await fetch("/api/edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, subject, target_new, ground_truth, method })
      });
      const data = await res.json();

      const pPreT = (data.pre_edit_target_prob * 100).toFixed(2);
      const pPostT = (data.post_edit_target_prob * 100).toFixed(2);
      const pPreOld = (data.pre_edit_old_prob * 100).toFixed(2);
      const pPostOld = (data.post_edit_old_prob * 100).toFixed(2);

      probContainer.innerHTML = `
        <div class="prob-card">
          <h3>✅ Edit Result (${data.editor.toUpperCase()})</h3>
          <p><strong>Latency:</strong> ${(data.execution_time_sec * 1000).toFixed(1)} ms &bull; <strong>Parameter Delta:</strong> &Delta;W = ${data.delta_norm}</p>
          <hr style="border-color:#45475a; margin:12px 0;">
          
          <p><strong>Target ("${target_new}") Probability:</strong></p>
          <div style="display:flex; justify-content:space-between; font-size:12px;"><span>Pre: ${pPreT}%</span><span>Post: ${pPostT}%</span></div>
          <div class="prob-bar">
            <div class="prob-fill target" style="width: ${Math.min(100, Math.max(5, data.post_edit_target_prob * 100))}%"></div>
          </div>

          <p><strong>Old Fact ("${ground_truth}") Probability:</strong></p>
          <div style="display:flex; justify-content:space-between; font-size:12px;"><span>Pre: ${pPreOld}%</span><span>Post: ${pPostOld}%</span></div>
          <div class="prob-bar">
            <div class="prob-fill ground-truth" style="width: ${Math.min(100, Math.max(5, data.post_edit_old_prob * 100))}%"></div>
          </div>
        </div>
      `;
      document.querySelector('[data-tab="tab-probs"]').click();
    } catch (err) {
      probContainer.innerHTML = `<p style="color:#f38ba8;">Error applying edit: ${err.message}</p>`;
    }
  });

  // Compare Button
  document.getElementById("btn-compare").addEventListener("click", async () => {
    const prompt = document.getElementById("edit-prompt").value;
    const subject = document.getElementById("subject-input").value;
    const target_new = document.getElementById("target-input").value;
    const ground_truth = document.getElementById("ground-truth-input").value;

    const radarContainer = document.getElementById("radar-container");
    const tableContainer = document.getElementById("comparison-table-container");

    radarContainer.innerHTML = `<p style="color:#89b4fa;">Benchmarking paradigms across Efficacy, Generality, and Locality...</p>`;
    tableContainer.innerHTML = "";

    try {
      const res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, subject, target_new, ground_truth, methods: ["rome", "memit", "pmet", "alphaedit", "grace", "ike"] })
      });
      const data = await res.json();

      radarContainer.innerHTML = data.svg_radar;

      let tableHtml = `
        <table class="table-styled">
          <thead>
            <tr>
              <th>Method</th>
              <th>Efficacy</th>
              <th>Generality</th>
              <th>Locality</th>
              <th>Portability</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
      `;

      for (const [method, m] of Object.entries(data.comparison)) {
        tableHtml += `
          <tr>
            <td><strong>${method.toUpperCase()}</strong></td>
            <td>${(m.efficacy * 100).toFixed(1)}%</td>
            <td>${(m.generality * 100).toFixed(1)}%</td>
            <td>${(m.locality * 100).toFixed(1)}%</td>
            <td>${(m.portability * 100).toFixed(1)}%</td>
            <td>${m.edit_latency_ms.toFixed(1)}ms</td>
          </tr>
        `;
      }
      tableHtml += `</tbody></table>`;
      tableContainer.innerHTML = tableHtml;
      document.querySelector('[data-tab="tab-radar"]').click();
    } catch (err) {
      radarContainer.innerHTML = `<p style="color:#f38ba8;">Error comparing methods: ${err.message}</p>`;
    }
  });

  // Check Conflict Button
  document.getElementById("btn-check-conflict").addEventListener("click", async () => {
    const prompt = document.getElementById("edit-prompt").value;
    const subject = document.getElementById("subject-input").value;
    const target_new = document.getElementById("target-input").value;

    try {
      const res = await fetch("/api/check-conflict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, subject, target_new })
      });
      const data = await res.json();
      if (data.has_conflict) {
        alert(`⚠️ Knowledge Conflict Detected!\nType: ${data.conflict_type}\nExplanation: ${data.explanation}`);
      } else {
        alert("✅ Knowledge Consistency Verified: No contradictions or dependency cycles detected.");
      }
    } catch (err) {
      alert(`Error checking conflict: ${err.message}`);
    }
  });

  // Continual Stream Runner
  document.getElementById("btn-run-continual").addEventListener("click", async () => {
    const contContainer = document.getElementById("continual-container");
    contContainer.innerHTML = `<p style="color:#89b4fa;">Executing lifelong sequential stream of 4 factual edits...</p>`;

    const sampleRequests = [
      { prompt: "The Eiffel Tower is in", target_new: "Rome", ground_truth: "Paris", subject: "Eiffel Tower" },
      { prompt: "Messi plays for", target_new: "Miami", ground_truth: "PSG", subject: "Messi" },
      { prompt: "The author of Hamlet was", target_new: "Cook", ground_truth: "Shakespeare", subject: "Hamlet" },
      { prompt: "Python was created by", target_new: "Musk", ground_truth: "Guido", subject: "Python" }
    ];

    try {
      const res = await fetch("/api/continual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ editor: "grace", requests: sampleRequests })
      });
      const data = await res.json();

      contContainer.innerHTML = `
        <div style="background:#181825; padding:12px; border-radius:8px; border:1px solid #45475a; margin-bottom:12px;">
          <strong>Average Lifelong Retention:</strong> ${(data.trajectory.average_retention * 100).toFixed(1)}% &bull; 
          <strong>Catastrophic Forgetting Rate:</strong> ${(data.trajectory.catastrophic_forgetting_rate * 100).toFixed(1)}% &bull;
          <strong>Wall-Clock Time:</strong> ${data.trajectory.wall_clock_time_sec}s
        </div>
        <div style="display:flex; justify-content:center;">${data.svg_matrix}</div>
      `;
    } catch (err) {
      contContainer.innerHTML = `<p style="color:#f38ba8;">Error running continual stream: ${err.message}</p>`;
    }
  });

  // Unlearn Button
  document.getElementById("btn-unlearn").addEventListener("click", async () => {
    const prompt = document.getElementById("unlearn-prompt").value;
    const target_to_erase = document.getElementById("unlearn-target").value;
    const resultContainer = document.getElementById("unlearn-results");

    resultContainer.innerHTML = `<p style="color:#89b4fa;">Executing machine unlearning on sensitive memory...</p>`;

    try {
      const res = await fetch("/api/unlearn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, target_to_erase, editor: "grace" })
      });
      const data = await res.json();

      resultContainer.innerHTML = `
        <div class="prob-card">
          <h4>🚫 Machine Unlearning Successful</h4>
          <p><strong>Erased Fact Target:</strong> "${data.target_erased}"</p>
          <p><strong>Probability Reduction:</strong> ${(data.probability_reduction * 100).toFixed(1)}%</p>
          <p><strong>Pre-Unlearn P("${data.target_erased}"):</strong> ${(data.pre_prob * 100).toFixed(3)}%</p>
          <p><strong>Post-Unlearn P("${data.target_erased}"):</strong> ${(data.post_prob * 100).toFixed(3)}%</p>
        </div>
      `;
    } catch (err) {
      resultContainer.innerHTML = `<p style="color:#f38ba8;">Error unlearning fact: ${err.message}</p>`;
    }
  });

  // Multi-Hop Evaluate Button
  document.getElementById("btn-run-multihop").addEventListener("click", async () => {
    const resultsContainer = document.getElementById("multihop-results");
    resultsContainer.innerHTML = `<p style="color:#89b4fa;">Evaluating multi-hop ripple propagation across reasoning chains...</p>`;

    try {
      const res = await fetch("/api/multihop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          anchor_prompt: "Lionel Messi plays for",
          anchor_subject: "Lionel Messi",
          anchor_relation: "plays for",
          anchor_target: "Inter Miami",
          downstream_queries: [
            {
              prompt: "The home stadium of the team Lionel Messi plays for is",
              target: "Chase Stadium",
              expected_previous: "Parc des Princes",
              hop_depth: 2
            },
            {
              prompt: "The league that the team Lionel Messi plays for competes in is",
              target: "Major League Soccer",
              expected_previous: "Ligue 1",
              hop_depth: 2
            }
          ],
          method: "grace"
        })
      });
      const data = await res.json();

      let hopRows = data.hop_results.map(h => `
        <div style="background:#1e1e2e; padding:8px 12px; margin-bottom:6px; border-radius:6px; border-left:3px solid ${h.correct ? '#a6e3a1' : '#f38ba8'};">
          <div style="font-weight:600; font-size:13px;">${h.prompt}</div>
          <div style="font-size:12px; color:#cdd6f4;">Predicted: <code>"${h.predicted}"</code> &bull; Expected: <code>"${h.target}"</code> &bull; Stale: <code>"${h.expected_previous}"</code></div>
          <div style="font-size:11px; color:${h.correct ? '#a6e3a1' : '#f9e2af'};">${h.correct ? '✅ Logically Propagated' : (h.retained_stale_knowledge ? '⚠️ Stale Memory Leakage' : '❌ Unresolved')}</div>
        </div>
      `).join("");

      resultsContainer.innerHTML = `
        <div class="prob-card">
          <h4>🔗 Multi-Hop Reasoning Assessment</h4>
          <p><strong>Anchor Prediction:</strong> "${data.anchor_predicted}" &bull; <strong>Anchor Efficacy:</strong> ${(data.anchor_efficacy * 100).toFixed(0)}%</p>
          <p><strong>Compositionality Score:</strong> ${(data.compositionality_score * 100).toFixed(1)}% &bull; <strong>Logical Soundness:</strong> ${(data.logical_soundness * 100).toFixed(1)}%</p>
          <hr style="border-color:#45475a; margin:10px 0;">
          <h5 style="margin-bottom:8px;">Downstream Multi-Hop Inferences:</h5>
          ${hopRows}
        </div>
      `;
    } catch (err) {
      resultsContainer.innerHTML = `<p style="color:#f38ba8;">Error evaluating multi-hop: ${err.message}</p>`;
    }
  });

  // Audit Model Drift Button
  document.getElementById("btn-run-audit").addEventListener("click", async () => {
    const resultsContainer = document.getElementById("multihop-results");
    resultsContainer.innerHTML = `<p style="color:#89b4fa;">Auditing general language capability retention and KL-divergence...</p>`;

    try {
      const res = await fetch("/api/audit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          max_allowed_kl: 1.25,
          max_allowed_ppl_ratio: 1.40
        })
      });
      const data = await res.json();
      const verdictColor = data.verdict === "PASS" ? "#a6e3a1" : (data.verdict === "WARNING" ? "#f9e2af" : "#f38ba8");

      resultsContainer.innerHTML = `
        <div class="prob-card">
          <h4>🛡️ Safety & Capability Audit Report</h4>
          <div style="display:inline-block; padding:4px 12px; border-radius:12px; font-weight:bold; font-size:12px; background:${verdictColor}22; color:${verdictColor}; border:1px solid ${verdictColor}; margin-bottom:12px;">
            VERDICT: ${data.verdict}
          </div>
          <p><strong>Perplexity Pre:</strong> ${data.pre_edit_perplexity} &bull; <strong>Perplexity Post:</strong> ${data.post_edit_perplexity} (Ratio: ${data.perplexity_drift_ratio})</p>
          <p><strong>Distributional KL Divergence:</strong> ${data.mean_kl_divergence} &bull; <strong>Min Token Entropy:</strong> ${data.min_token_entropy}</p>
          <p><strong>Entropy Collapse Detected:</strong> ${data.entropy_collapse_detected ? '🚨 YES' : '✅ NO'}</p>
          ${data.violations.length ? `<p style="color:#f38ba8;"><strong>Violations:</strong> ${data.violations.join('; ')}</p>` : ''}
        </div>
      `;
    } catch (err) {
      resultsContainer.innerHTML = `<p style="color:#f38ba8;">Error auditing model: ${err.message}</p>`;
    }
  });

  // Rollback Button
  document.getElementById("btn-rollback").addEventListener("click", async () => {
    const resultsContainer = document.getElementById("multihop-results");
    try {
      const res = await fetch("/api/rollback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rollback_all: false })
      });
      const data = await res.json();
      resultsContainer.innerHTML = `
        <div class="prob-card">
          <h4>⏮️ Rollback Operation</h4>
          <p><strong>Status:</strong> ${data.status}</p>
          <p><strong>Message / Patch:</strong> ${data.message || data.rolled_back_patch || 'Completed'}</p>
        </div>
      `;
    } catch (err) {
      resultsContainer.innerHTML = `<p style="color:#f38ba8;">Error executing rollback: ${err.message}</p>`;
    }
  });
});
