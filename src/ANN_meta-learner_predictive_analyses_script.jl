using Flux, Flux.Zygote, MLJ, CSV, DataFrames, Statistics, BSON, CairoMakie, Jitterplot;
cd(@__DIR__);
df = DataFrame(CSV.File("../data/H2O2_ML_data.csv"));
df = df[:, [4, 7, 8, 9, 10, 11, 12, 13, 14]];
pred_feat_cathode_to_anode_ratio = DataFrame(cathode_to_anode_ratio = unique(df.cathode_to_anode_ratio));
pred_feat_substrate = DataFrame(substrate = unique(df.substrate));
pred_feat_applied_voltage = DataFrame(applied_voltage = unique(df.applied_voltage));
pred_feat_anode_electrode = DataFrame(anode_electrode = unique(df.anode_electrode));
pred_feat_cathode_electrode = DataFrame(cathode_electrode = unique(df.cathode_electrode));
pred_feat_cathode_operating_mode = DataFrame(cathode_operating_mode = unique(df.cathode_operating_mode));
pred_feat_separator = DataFrame(separator = unique(df.separator));
pred_df = crossjoin(pred_feat_cathode_to_anode_ratio, pred_feat_substrate, pred_feat_applied_voltage, pred_feat_anode_electrode, pred_feat_cathode_electrode, pred_feat_cathode_operating_mode, pred_feat_separator);
pred_df[!, :total_volume] .= 1.0;
pred_df = pred_df[:, Symbol.(names(df))[1:(end-1)]];
show(first(pred_df, 10), allcols = true)
for cols in 3:7
	pred_df[!, cols] .= MLJ.categorical(pred_df[:, cols])
end;
pred_df0 = pred_df;
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, pred_df));
pred_df = MLJ.transform(mach, pred_df);
show(first(pred_df, 10), allcols = true)
x = pred_df[:, 1:26];
show(first(x, 10), allcols = true)
J, N = size(x);

# Deep neural network with 4 hidden layers (DeepNet4HL)
mutable struct nnet_mod_builder4
    n1 :: Int
    n2 :: Int
    n3 :: Int
    n4 :: Int
end;
function nnet_build(nn :: nnet_mod_builder4, n_in, n_out)
    return Flux.Chain(Dense(n_in, nn.n1, Flux.NNlib.relu, init = Flux.kaiming_normal),
                 Dense(nn.n1, nn.n2, Flux.NNlib.relu, init = Flux.kaiming_normal),
                 Dense(nn.n2, nn.n3, Flux.NNlib.relu, init = Flux.kaiming_normal),
                 Dense(nn.n3, nn.n4, Flux.NNlib.relu, init = Flux.kaiming_normal),
                 Dense(nn.n4, n_out, init = Flux.kaiming_normal))
end;
meta_model = nnet_build(nnet_mod_builder4(N * 16, N * 12, N * 8, N * 4), N, 1);
meta_mod_3_fold = BSON.load("nnet_model_architechture_4_3-fold_CV_h2o2_production_rate_trained_model/saved_ensembled_meta_model(s)/trained_model.bson")[:flux_model];
meta_mod_5_fold = BSON.load("nnet_model_architechture_4_5-fold_CV_h2o2_production_rate_trained_model/saved_ensembled_meta_model(s)/trained_model.bson")[:flux_model];

for (p, q, r) in zip(Flux.params(meta_model), Flux.params(meta_mod_3_fold), Flux.params(meta_mod_3_fold))
	p .= (q .+ r) ./ 2.0
end;
pred_df0[!, :predicted_production_rate] .= vec(meta_model(Matrix(Matrix(x)')));
pred_df0 = pred_df0[pred_df0.predicted_production_rate .> 0, :];
sort!(pred_df0, :predicted_production_rate, rev = true);
df_flow = DataFrame(flowrate_gpd = [10000, 10000, 10000, 20000, 20000, 20000, 30000, 30000, 30000],
h2o2_concentration_mgpL = [100, 200, 300, 100, 200, 300, 100, 200, 300],
required_h2o2_productivity_kgpd = [3.8, 7.6, 11.4, 7.6, 15.2, 22.8, 11.4, 22.8, 34.2]);
pred_df0 = crossjoin(pred_df0, df_flow);
sort!(pred_df0, [:predicted_production_rate, :substrate], rev = true);
pred_df0[!, :required_number_of_reactors] .= 1.0 .+ div.(pred_df0.required_h2o2_productivity_kgpd,  pred_df0.predicted_production_rate);
sort!(pred_df0, [:required_number_of_reactors, :substrate], rev = false);
show(first(pred_df0, 10), allcols = true)
lookup_substrate = combine(groupby(pred_df0, [:substrate, :flowrate_gpd, :h2o2_concentration_mgpL]),
:required_number_of_reactors => (x -> minimum(x)) => :required_number_of_reactors);
pred_df00 = innerjoin(pred_df0, lookup_substrate, on = [:required_number_of_reactors, :substrate, :flowrate_gpd, :h2o2_concentration_mgpL]);
sort!(pred_df00, [:required_number_of_reactors, :substrate], rev = false);
# CSV.write("outputs/Table S2. Predicted values for different possible combinations of feature values.csv", pred_df00);

df_wastewater = pred_df00[pred_df00.substrate .== "Municipal Wastewater", :];
sort!(df_wastewater, [:flowrate_gpd, :h2o2_concentration_mgpL, :applied_voltage, :cathode_to_anode_ratio], rev = false);
# CSV.write("outputs/Table S3. Predicted values for different possible combinations of reactor parameters with municipal wastewater as substrate.csv", df_wastewater);
df_wastewater1 = combine(groupby(df_wastewater, [:anode_electrode, :cathode_electrode, :cathode_operating_mode, :separator, :flowrate_gpd, :h2o2_concentration_mgpL, :required_h2o2_productivity_kgpd]),
:cathode_to_anode_ratio => (x->extrema(x)[1]) => :min_cathode_to_anode_ratio,
:cathode_to_anode_ratio => (x->extrema(x)[2]) => :max_cathode_to_anode_ratio,
:applied_voltage => (x->extrema(x)[1]) => :min_applied_voltage,
:applied_voltage => (x->extrema(x)[2]) => :max_applied_voltage,
:predicted_production_rate => (x->extrema(x)[1]) => :min_predicted_production_rate,
:predicted_production_rate => (x->extrema(x)[2]) => :max_predicted_production_rate,
:required_number_of_reactors => (x->extrema(x)[1]) => :min_required_number_of_reactors,
:required_number_of_reactors => (x->extrema(x)[2]) => :max_required_number_of_reactors);

df_wastewater1[!, :range_cathode_to_anode_ratio] .= string.(df_wastewater1.min_cathode_to_anode_ratio) .* "-" .* string.(df_wastewater1.max_cathode_to_anode_ratio);
df_wastewater1[!, :range_applied_voltage] .= string.(df_wastewater1.min_applied_voltage) .* "-" .* string.(df_wastewater1.max_applied_voltage);
df_wastewater1[!, :range_predicted_production_rate] .= string.(round.(df_wastewater1.min_predicted_production_rate, digits = 2)) .* "-" .* string.(round.(df_wastewater1.max_predicted_production_rate, digits = 2));
df_wastewater1[!, :range_required_number_of_reactors] .= string.(df_wastewater1.min_required_number_of_reactors) .* "-" .* string.(df_wastewater1.max_required_number_of_reactors);
sort!(df_wastewater1, [:flowrate_gpd, :h2o2_concentration_mgpL], rev = false);

df_wastewater1 = df_wastewater1[:, [:flowrate_gpd, :h2o2_concentration_mgpL, :required_h2o2_productivity_kgpd, :anode_electrode, :cathode_electrode, :cathode_operating_mode, :separator, :range_cathode_to_anode_ratio, :range_applied_voltage, :range_predicted_production_rate, :min_required_number_of_reactors]];

CSV.write("outputs/Table 1. Predicted values for range of possible combinations of reactor parameters with municipal wastewater as substrate.csv", df_wastewater1);


df_graph = DataFrame();
for (i, substrate) in zip([1, 2, 3, 4, 5, 6], ["Acetate", "Glucose", "Lactate", "Sucrose", "Primary Sludge", "Municipal Wastewater"])
	df_graph0 = DataFrame(substrate_tag = i, substrate = substrate, substrate_flag = df[:, i], y_pred = df[:, Symbol("predicted H₂O₂ production rate")])
	append!(df_graph,df_graph0)
end;
df_graph = df_graph[df_graph.substrate_flag .== 1.0, :];
show(first(df_graph, 10), allcols = true)

df_plot1 = df;
df_plot1[!, :vol_class] .= ifelse.(df_plot1.total_volume .<= 0.0004, 1, ifelse.((df_plot1.total_volume .> 0.0004) .& (df_plot1.total_volume .<= 0.00068), 2, 3));
df_plot1[!, :volt_class] .= ifelse.(df_plot1.applied_voltage .< 0.2, 1, ifelse.((df_plot1.applied_voltage .>= 0.2) .& (df_plot1.applied_voltage .<= 0.7), 2, 3));
df_plot1[!, :ratio_class] .= ifelse.(df_plot1.cathode_to_anode_ratio .< 0.22, 1, ifelse.((df_plot1.cathode_to_anode_ratio .>= 0.22) .& (df_plot1.cathode_to_anode_ratio .<= 0.25), 2, 3));

# Jitterplot - Predicted H2O2 production

# H2O2 production rate

f = Makie.Figure(resolution = (800, 800))
ax = Makie.Axis(f[1, 1], xticklabelsvisible = true, yticklabelsvisible = true, xticks = (2:1:6, unique(df_graph.substrate)), yticks = [3.6, 7.6, 11.4, 15.2], xticklabelrotation = pi/2, ylabel = "Predicted H₂O₂ production rate (kg m⁻³ d⁻¹)", ylabelsize = 20, xlabel = "Substrate", xlabelsize = 20);
x_plot = df_graph.substrate_tag;
y_plot = df_graph.y_pred;
for i in unique(df_graph.substrate_tag)
	df_hist = df_graph[df_graph.substrate_tag .== i, :];
	hist!(ax, df_hist.y_pred, color = (:green, 0.5), bins = 30, normalization = :probability, scale_to=-0.6, offset=i, direction=:x)
end
Makie.boxplot!(x_plot, y_plot, show_notch = false, color = (:white, 0.0), strokewidth = 1, show_outliers = true);
f

ax = Makie.Axis(f[1, 2], xticklabelsvisible = true, yticklabelsvisible = true, yticks = [3.6, 7.6, 11.4, 15.2], xticklabelrotation = pi/2, ylabelsize = 20, xlabel = "Total volume", xlabelsize = 20, xticklabelsize = 12);
x_plot = df_plot1.vol_class;
y_plot = df_plot1[:, Symbol("predicted H₂O₂ production rate")];
for i in 1:length(unique(x_plot))
	df_hist = df_plot1[df_plot1.vol_class .== i, :];
	hist!(ax, df_hist[:, Symbol("predicted H₂O₂ production rate")], color = (:green, 0.5), bins = 30, normalization = :probability, scale_to=-0.6, offset=i, direction=:x)
end
Makie.boxplot!(x_plot, y_plot, width = 0.25, show_notch = false, color = (:white, 0.0), strokewidth = 1, show_outliers = true);
ax = Makie.Axis(f[2, 1], xticklabelsvisible = true, yticklabelsvisible = true, yticks = [3.6, 7.6, 11.4, 15.2], xticklabelrotation = pi/2, ylabel = "Predicted H₂O₂ production rate (kg m⁻³ d⁻¹)", ylabelsize = 20, xlabel = "Applied voltage", xlabelsize = 20, xticklabelsize = 12);
x_plot = df_plot1.volt_class;
y_plot = df_plot1[:, Symbol("predicted H₂O₂ production rate")];
Makie.boxplot!(x_plot, y_plot, width = 0.05, show_notch = false, color = (:white, 0.0), strokewidth = 1, show_outliers = true);
ax = Makie.Axis(f[3, 1], xticklabelsvisible = true, yticklabelsvisible = true, yticks = [3.6, 7.6, 11.4, 15.2], xticklabelrotation = pi/2, xlabel = "Cathode to anode ratio", xlabelsize = 20, xticklabelsize = 12);
x_plot = df_plot1.ratio_class;
y_plot = df_plot1[:, Symbol("predicted H₂O₂ production rate")];
Makie.boxplot!(x_plot, y_plot, width = 0.05, show_notch = false, color = (:white, 0.0), strokewidth = 1, show_outliers = true);
f



Makie.Label(f[1:2, 1], "Predicted H₂O₂ production rate (kg m⁻³ d⁻¹)", rotation = pi/2, textsize = 22);
ax = Makie.Axis(f[1, 2], title = "(a) ANN", titlealign = :left, titlesize = 26, xticklabelsvisible = false, yticklabelsvisible = true)
df_graph = df_pred1[df_pred1.model_tag .== 1, :];
x_plot = df.production_rate;
y_plot = df_graph.y_pred;
Makie.scatter!(x_plot, y_plot, color = :green, strokewidth = 1);
Makie.lines!([extrema(df_pred1.y_pred)[1], extrema(df_pred1.y_pred)[2]], [extrema(df_pred1.y_pred)[1], extrema(df_pred1.y_pred)[2]], color = :blue);
Makie.text!("R² = " * string(round(Statistics.cor(x_plot, y_plot)^2.0, digits = 3)) * "\nRMSE = " * string(round(sqrt(Flux.Losses.mse(y_plot, x_plot)), digits = 3)), color = :blue, position = (17, 1), align = [:right, :bottom]);

ax = Makie.Axis(f[1, 3], title = "(b) DeepNet2HL", titlealign = :left, titlesize = 26, xticklabelsvisible = false, yticklabelsvisible = false)
df_graph = df_pred1[df_pred1.model_tag .== 2, :];
x_plot = df.production_rate;
y_plot = df_graph.y_pred;
Makie.scatter!(x_plot, y_plot, color = :green, strokewidth = 1);
Makie.lines!([extrema(df_pred1.y_pred)[1], extrema(df_pred1.y_pred)[2]], [extrema(df_pred1.y_pred)[1], extrema(df_pred1.y_pred)[2]], color = :blue);
Makie.text!("R² = " * string(round(Statistics.cor(x_plot, y_plot)^2.0, digits = 3)) * "\nRMSE = " * string(round(sqrt(Flux.Losses.mse(y_plot, x_plot)), digits = 3)), color = :blue, position = (17, 1), align = [:right, :bottom]);

ax = Makie.Axis(f[2, 2], title = "(c) DeepNet3HL", titlealign = :left, titlesize = 26, xticklabelsvisible = true, yticklabelsvisible = true)
df_graph = df_pred1[df_pred1.model_tag .== 3, :];
x_plot = df.production_rate;
y_plot = df_graph.y_pred;
Makie.scatter!(x_plot, y_plot, color = :green, strokewidth = 1);
Makie.lines!([extrema(df_pred1.y_pred)[1], extrema(df_pred1.y_pred)[2]], [extrema(df_pred1.y_pred)[1], extrema(df_pred1.y_pred)[2]], color = :blue);
Makie.text!("R² = " * string(round(Statistics.cor(x_plot, y_plot)^2.0, digits = 3)) * "\nRMSE = " * string(round(sqrt(Flux.Losses.mse(y_plot, x_plot)), digits = 3)), color = :blue, position = (17, 1), align = [:right, :bottom]);

ax = Makie.Axis(f[2, 3], title = "(d) DeepNet4HL", titlealign = :left, titlesize = 26, xticklabelsvisible = true, yticklabelsvisible = false)
df_graph = df_pred1[df_pred1.model_tag .== 4, :];
x_plot = df.production_rate;
y_plot = df_graph.y_pred;
Makie.scatter!(x_plot, y_plot, color = :green, strokewidth = 1);
Makie.lines!([extrema(df_pred1.y_pred)[1], extrema(df_pred1.y_pred)[2]], [extrema(df_pred1.y_pred)[1], extrema(df_pred1.y_pred)[2]], color = :blue);
Makie.text!("R² = " * string(round(Statistics.cor(x_plot, y_plot)^2.0, digits = 3)) * "\nRMSE = " * string(round(sqrt(Flux.Losses.mse(y_plot, x_plot)), digits = 3)), color = :blue, position = (17, 1), align = [:right, :bottom]);
Makie.Label(f[3, 1:3], "Reported H₂O₂ production rate (kg m⁻³ d⁻¹)", textsize = 22);
f
Makie.save("outputs/Fig. 3. Final ensemble model performance scatterplots for H2O2 production rate.png", px_per_unit = 3.3, f);
