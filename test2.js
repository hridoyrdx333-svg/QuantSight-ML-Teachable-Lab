
    const heavyButtons = [
        "btn-collect",
        "btn-validate",
        "btn-features",
        "btn-train",
        "btn-promote",
        "btn-export"
    ];


    function metric(value, digits = 4) {
        if (
            value === null
            || value === undefined
            || Number.isNaN(Number(value))
        ) {
            return "-";
        }

        return Number(value).toFixed(digits);
    }


    function rows(item) {
        if (!item || !item.exists) {
            return "-";
        }

        return Number(
            item.rows || 0
        ).toLocaleString();
    }


    function duration(seconds) {
        seconds = Number(seconds || 0);

        const h =
            Math.floor(seconds / 3600);

        const m =
            Math.floor(
                (seconds % 3600) / 60
            );

        const s =
            Math.floor(seconds % 60);

        return [
            String(h).padStart(2, "0"),
            String(m).padStart(2, "0"),
            String(s).padStart(2, "0")
        ].join(":");
    }


    function disableHeavy(disabled) {
        heavyButtons.forEach(id => {
            const element =
                document.getElementById(id);

            if (element) {
                element.disabled =
                    disabled;
            }
        });
    }


    function readinessLabel(value) {
        return value
            ? '<span class="good">PASS</span>'
            : '<span class="bad">NOT READY</span>';
    }


    async function refresh() {
        try {
            const response =
                await fetch(
                    "/api/status",
                    {
                        cache: "no-store"
                    }
                );

            if (!response.ok) {
                throw new Error(
                    "Status API HTTP "
                    + response.status
                );
            }

            const data =
                await response.json();


            document.getElementById(
                "massive"
            ).textContent =
                data.massive_api_key
                    ? "CONNECTED"
                    : "MISSING";


            const validation =
                data.validation || {};

            document.getElementById(
                "validation"
            ).textContent =
                validation.overall
                || validation.status
                || (
                    Object.keys(validation).length
                        ? "CHECK"
                        : "None"
                );


            const features =
                data.features || {};

            document.getElementById(
                "feature-version"
            ).textContent =
                features.current_version
                || "None";


            const approved =
                data.approved || {};

            document.getElementById(
                "approved"
            ).textContent =
                approved.model_name
                || approved.model
                || "None";


            const task =
                data.task || {};

            const running =
                Boolean(task.running);

            document.getElementById(
                "current-task"
            ).textContent =
                running
                    ? (
                        task.label
                        || "Running"
                    )
                    : "Idle";

            document.getElementById(
                "task-status"
            ).textContent =
                task.status
                || "idle";

            document.getElementById(
                "task-stage"
            ).textContent =
                task.stage
                || "-";

            document.getElementById(
                "task-pid"
            ).textContent =
                task.pid
                || "-";

            document.getElementById(
                "task-elapsed"
            ).textContent =
                duration(
                    task.elapsed_seconds
                );

            document.getElementById(
                "task-silent"
            ).textContent =
                Number(
                    task.silent_seconds
                    || 0
                ).toLocaleString()
                + " sec";

            document.getElementById(
                "task-exit"
            ).textContent =
                task.exit_code === null
                || task.exit_code === undefined
                    ? "-"
                    : task.exit_code;

            document.getElementById(
                "latest-result"
            ).textContent =
                task.last_message
                || "Ready";


            disableHeavy(
                running
            );

            document.getElementById(
                "cancel-task"
            ).disabled =
                !running;


            const banner =
                document.getElementById(
                    "stall-banner"
                );

            banner.className =
                "banner";

            banner.textContent =
                "";

            if (
                task.stall_level
                === "warning"
            ) {
                banner.className =
                    "banner banner-warning";

                banner.textContent =
                    "Warning: 10 মিনিট ধরে নতুন output নেই।";
            }

            if (
                task.stall_level
                === "critical"
            ) {
                banner.className =
                    "banner banner-critical";

                banner.textContent =
                    "Critical: 20 মিনিট ধরে progress নেই। Cancel করে investigate করুন।";
            }


            const live =
                data.live || {};

            const liveRunning =
                Boolean(
                    live.running
                );

            const liveStatus = document.getElementById("live-status");
            liveStatus.textContent = liveRunning ? ("RUNNING" + (live.pid ? " · " + live.pid : "")) : "STOPPED";
            liveStatus.className = "value " + (liveRunning ? "running" : "stopped");

            const builder = data.builder || {};
            const builderRunning = Boolean(builder.running);
            const builderStatus = document.getElementById("builder-status");
            builderStatus.textContent = builderRunning ? ("RUNNING" + (builder.pid ? " · " + builder.pid : "")) : "STOPPED";
            builderStatus.className = "value " + (builderRunning ? "running" : "stopped");

            const shadowProc = data.shadow_proc || {};
            const shadowProcRunning = Boolean(shadowProc.running);
            const shadowProcStatus = document.getElementById("shadow-proc-status");
            shadowProcStatus.textContent = shadowProcRunning ? ("RUNNING" + (shadowProc.pid ? " · " + shadowProc.pid : "")) : "STOPPED";
            shadowProcStatus.className = "value " + (shadowProcRunning ? "running" : "stopped");

            document.getElementById(
                "start-live"
            ).disabled =
                liveRunning;

            document.getElementById(
                "stop-live"
            ).disabled =
                !liveRunning;


            const progress =
                data.progress || {};

            document.getElementById(
                "v4-progress"
            ).textContent =
                (
                    progress.percent
                    ?? 0
                )
                + "%";


            const shadow =
                data.shadow || {};

            const shadowLatest =
                shadow.latest || {};

            const shadowFresh =
                Boolean(
                    shadow.fresh
                );

            const shadowNoErrors =
                Boolean(
                    shadow.no_errors
                );

            const shadowTop =
                document.getElementById(
                    "shadow-status"
                );

            if (shadowTop) {
                shadowTop.textContent =
                    shadowFresh
                        ? "FRESH"
                        : (
                            shadowNoErrors
                                ? "STALE"
                                : "WAITING"
                        );

                shadowTop.className =
                    "value "
                    + (
                        shadowFresh
                            ? "running"
                            : "stopped"
                    );
            }

            const shadowBody =
                document.getElementById(
                    "shadow-body"
                );

            shadowBody.innerHTML = "";

            [
                "BTCUSDT",
                "ETHUSDT",
                "SOLUSDT"
            ].forEach(
                symbol => {
                    const item =
                        shadowLatest[
                            symbol
                        ] || {};

                    const decision =
                        item.decision
                        || "-";

                    const probability =
                        item.probability_up;

                    const threshold =
                        item.locked_threshold;

                    const featureAge =
                        item.feature_age_seconds;

                    const tr =
                        document.createElement(
                            "tr"
                        );

                    tr.innerHTML = `
                        <td>${symbol}</td>
                        <td>${decision}</td>
                        <td>${metric(probability)}</td>
                        <td>${metric(threshold, 2)}</td>
                        <td>${
                            featureAge === null
                            || featureAge === undefined
                                ? "-"
                                : Math.round(
                                    Number(
                                        featureAge
                                    )
                                ) + " sec"
                        }</td>
                        <td>${
                            item.orders_sent === true
                                ? "YES"
                                : "NO"
                        }</td>
                    `;

                    shadowBody.appendChild(
                        tr
                    );
                }
            );

            const shadowRunnerStatus =
                (
                    shadow.status
                    || {}
                ).status
                || "not running";

            const builderNoteStatus =
                (
                    shadow.feature_builder
                    || {}
                ).status
                || "unknown";

            document.getElementById(
                "shadow-note"
            ).textContent =
                "Runner: "
                + shadowRunnerStatus
                + " · Feature builder: "
                + builderNoteStatus
                + " · Fresh: "
                + (
                    shadowFresh
                        ? "YES"
                        : "NO"
                );


            const readiness =
                data.readiness || {};

            const readinessBody =
                document.getElementById(
                    "readiness-body"
                );

            readinessBody.innerHTML = "";

            const readinessItems = {
                "Historical Data":
                    readiness.historical_data,

                "Funding Data":
                    readiness.funding_data,

                "Orderbook Collection Started":
                    readiness.orderbook_collection_started,

                "Feature Build":
                    readiness.feature_build,

                "Validation Report":
                    readiness.validation_report,

                "Training Report":
                    readiness.training_report,

                "Approved Model":
                    readiness.approved_model,

                "Live Shadow Fresh":
                    readiness.live_shadow_fresh,

                "Paper Trade Ready":
                    readiness.paper_trade_ready
            };

            Object.entries(
                readinessItems
            ).forEach(
                ([name, value]) => {
                    const tr =
                        document.createElement(
                            "tr"
                        );

                    tr.innerHTML = `
                        <td>${name}</td>
                        <td>${readinessLabel(value)}</td>
                    `;

                    readinessBody.appendChild(
                        tr
                    );
                }
            );


            const modelData =
                data.models || {};

            const reports =
                modelData.reports || [];

            const modelBody =
                document.getElementById(
                    "model-body"
                );

            modelBody.innerHTML = "";

            let modelRows = 0;

            reports.forEach(
                report => {

                    const models =
                        report.models || {};

                    Object.entries(
                        models
                    ).forEach(
                        ([name, result]) => {

                            const test =
                                result.test || {};

                            const tr =
                                document.createElement(
                                    "tr"
                                );

                            tr.innerHTML = `
                                <td>${report.feature_version || "-"}</td>
                                <td>${name}</td>
                                <td>${metric(test.roc_auc)}</td>
                                <td>${metric(test.precision)}</td>
                                <td>${metric(test.recall)}</td>
                                <td>${metric(test.brier)}</td>
                                <td>${metric(test.log_loss)}</td>
                            `;

                            modelBody.appendChild(
                                tr
                            );

                            modelRows++;
                        }
                    );
                }
            );

            document.getElementById(
                "model-summary"
            ).textContent =
                modelRows
                    ? (
                        reports.length
                        + " saved feature-version report(s) found."
                    )
                    : (
                        "No saved training report available."
                    );


            const featureBody =
                document.getElementById(
                    "feature-body"
                );

            featureBody.innerHTML = "";

            (
                features.versions
                || []
            ).forEach(
                version => {

                    const tr =
                        document.createElement(
                            "tr"
                        );

                    tr.innerHTML = `
                        <td>${version.version || "-"}</td>
                        <td>${version.source_timeframe || "-"}</td>
                        <td>${version.files || 0}</td>
                        <td>${
                            version.orderbook_in_training === true
                                ? "YES"
                                : version.orderbook_in_training === false
                                    ? "NO"
                                    : "-"
                        }</td>
                    `;

                    featureBody.appendChild(
                        tr
                    );
                }
            );


            const datasetBody =
                document.getElementById(
                    "dataset-body"
                );

            datasetBody.innerHTML = "";

            const dataset =
                data.rows || {};

            Object.keys(
                dataset
            ).forEach(
                symbol => {

                    const item =
                        dataset[symbol];

                    const ob =
                        item.orderbook || {};

                    const coverage =
                        Number(
                            ob.coverage_hours
                            || 0
                        );

                    const coverageText =
                        coverage >= 24
                            ? (
                                (coverage / 24)
                                .toFixed(2)
                                + " days"
                            )
                            : (
                                coverage.toFixed(2)
                                + " h"
                            );

                    const tr =
                        document.createElement(
                            "tr"
                        );

                    tr.innerHTML = `
                        <td>${symbol}</td>
                        <td>${rows(item["1m"])}</td>
                        <td>${rows(item["5m"])}</td>
                        <td>${rows(item["15m"])}</td>
                        <td>${rows(item["1h"])}</td>
                        <td>${rows(item["funding"])}</td>
                        <td>${rows(item["orderbook"])}</td>
                        <td>${coverageText}</td>
                    `;

                    datasetBody.appendChild(
                        tr
                    );
                }
            );

        } catch (error) {
            document.getElementById(
                "latest-result"
            ).textContent =
                "UI refresh error: "
                + error
                + "\nCheck that ML_LAB.bat / ui/app.py is still running on port 8765.";
        }
    }


    async function action(name) {
        try {
            const response =
                await fetch(
                    "/api/action/" + name,
                    {
                        method: "POST"
                    }
                );

            const data =
                await response.json();

            if (!data.ok) {
                alert(
                    data.message
                    || "Action failed"
                );
            }

            await refresh();

        } catch (error) {
            alert(error);
        }
    }


    async function cancelTask() {
        if (
            !confirm(
                "Current heavy task cancel করবেন?"
            )
        ) {
            return;
        }

        const response =
            await fetch(
                "/api/task/cancel",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (!data.ok) {
            alert(
                data.message
            );
        }

        await refresh();
    }


    async function startLive() {
        const response =
            await fetch(
                "/api/live/start",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (!data.ok) {
            alert(
                data.message
            );
        }

        await refresh();
    }


    async function stopLive() {
        const response =
            await fetch(
                "/api/live/stop",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (!data.ok) {
            alert(
                data.message
            );
        }

        await refresh();
    }


    async function promoteModel() {
        const model =
            prompt(
                "Model name:\n"
                + "logistic\n"
                + "random_forest\n"
                + "hist_gradient_boosting"
            );

        if (!model) {
            return;
        }

        const response =
            await fetch(
                "/api/promote",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify(
                        {
                            model:
                                model.trim()
                        }
                    )
                }
            );

        const data =
            await response.json();

        if (!data.ok) {
            alert(
                data.message
            );
        }

        await refresh();
    }


    refresh();

    setInterval(
        refresh,
        2000
    );
