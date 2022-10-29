using Flux, Flux.Zygote, MLJ, CSV, DataFrames, Statistics, ShapML, BSON, CairoMakie, Jitterplot;
cd(@__DIR__);
# Predefined functions
function predict_function(model, data :: DataFrame)
  data_pred = DataFrame(y_pred = vec(model(Matrix(data)')))
  return data_pred
end;
function normalize(x)
    return (x .- extrema(x)[1])/(extrema(x)[2] - extrema(x)[1])
end;
function standardize_(x)
    return (x .- mean(x))/sqrt(var(x))
end;
function destandardize_(conv_x, mean_x, var_x)
	return conv_x .* sqrt(var_x) .+ mean_x
end;
df = DataFrame(CSV.File("../data/H2O2_ML_data.csv"));
df = df[:, [4, 7, 8, 9, 10, 11, 12, 13, 14]];
show(first(df, 10), allcols = true)
for cols in 3:7
	df[!, cols] .= MLJ.categorical(df[:, cols])
end;
hot = MLJ.OneHotEncoder(ordered_factor = true);
mach = MLJ.fit!(machine(hot, df));
df = MLJ.transform(mach, df);
show(first(df, 10), allcols = true)

x = df[:, 1:26];
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

# feature importance shapley values
shap_df = ShapML.shap(explain = x,
                        reference = x,
                        model = meta_model,
                        predict_function = predict_function,
                        sample_size = 60,
                        seed = 1);
show(first(shap_df, 10), allcols = true);
graph_feat_lab_df = DataFrame(
revised_labels = ["Total Volume (m³)",
"Cathode/Anode",
"Substrate: Acetate\n(No = 0, Yes = 1)",
"Substrate: Glucose\n(No = 0, Yes = 1)",
"Substrate: Lactate\n(No = 0, Yes = 1)",
"Substrate: Municipal Wastewater\n(No = 0, Yes = 1)",
"Substrate: Primary Sludge\n(No = 0, Yes = 1)",
"Substrate: Sucrose\n(No = 0, Yes = 1)",
"Anode Electrode: Carbon Brush\n(No = 0, Yes = 1)",
"Anode Electrode: Carbon Felt\n(No = 0, Yes = 1)",
"Anode Electrode: Carbon Fiber\n(No = 0, Yes = 1)",
"Anode Electrode: Graphite\n(No = 0, Yes = 1)",
"Anode Electrode: Surface Modified/Composite Electrode\n(No = 0, Yes = 1)",
"Cathode Electrode: Activated Carbon\n(No = 0, Yes = 1)",
"Cathode Electrode: Carbon Black\n(No = 0, Yes = 1)",
"Cathode Electrode: Carbon Cloth\n(No = 0, Yes = 1)",
"Cathode Electrode: Carbon Felt\n(No = 0, Yes = 1)",
"Cathode Electrode: GDE\n(No = 0, Yes = 1)",
"Cathode Electrode: Graphene\n(No = 0, Yes = 1)",
"Cathode Electrode: Graphite\n(No = 0, Yes = 1)",
"Cathode Electrode: Surface Modified/Composite Electrode\n(No = 0, Yes = 1)",
"Cathode Operating Mode: Batch\n(No = 0, Yes = 1)",
"Cathode Operating Mode: Continuous\n(No = 0, Yes = 1)",
"Separator: AEM\n(No = 0, Yes = 1)",
"Separator: CEM\n(No = 0, Yes = 1)",
"Applied Voltage (V)"],
feature_name = ["total_volume",
"cathode_to_anode_ratio",
"substrate__Acetate",
"substrate__Glucose",
"substrate__Lactate",
"substrate__Municipal Wastewater",
"substrate__Primary Sludge",
"substrate__Sucrose",
"anode_electrode__Carbon Brush",
"anode_electrode__Carbon Felt",
"anode_electrode__Carbon Fiber",
"anode_electrode__Graphite",
"anode_electrode__Surface Modified/Composite Electrode",
"cathode_electrode__Activated Carbon",
"cathode_electrode__Carbon Black ",
"cathode_electrode__Carbon Cloth",
"cathode_electrode__Carbon Felt",
"cathode_electrode__GDE",
"cathode_electrode__Graphene",
"cathode_electrode__Graphite",
"cathode_electrode__Surface Modified/Composite Electrode",
"cathode_operating_mode__Batch",
"cathode_operating_mode__Continuous",
"separator__AEM",
"separator__CEM",
"applied_voltage"]);

shap_df_sum0 = combine(groupby(shap_df, :feature_name),
:shap_effect => (x -> mean(abs.(x))) => :abs_shap_values);
shap_df_sum0 = innerjoin(shap_df_sum0, graph_feat_lab_df, on = :feature_name);
sort!(shap_df_sum0, [:abs_shap_values], rev = true);
shap_df_sum0[!, :feature_tag] .= length(unique(shap_df_sum0.feature_name)) : -1 : 1;
show(first(shap_df_sum0, 10), allcols = true)

shap_df_sum = innerjoin(shap_df_sum0, shap_df, on = :feature_name);
sort!(shap_df_sum, [:abs_shap_values], rev = true);
show(first(shap_df_sum, 10), allcols = true)

# Plotting shap values
color_map = Makie.cgrad([:green, :orange, :red], alpha = 0.6);
color_map1 = Makie.cgrad([:green, :orange, :red], alpha = 1.0);

# H2O2 production rate
f = Makie.Figure(resolution = (2200, 2200));
ax = Makie.Axis(f[1, 1], title = "(a)", yticks = (shap_df_sum0.feature_tag, shap_df_sum0.revised_labels), xlabel = "Mean absolute shapely value", ylabel = "Features for predicting H₂O₂ production rate (kg m⁻³ d⁻¹)", xlabelsize = 42, ylabelsize = 42, yticksize = 12, yticklabelsize = 26, xticklabelsize = 30, titlesize = 50, titlealign = :left);
ylims!(0.25, length(unique(shap_df_sum.feature_name)) + 0.75);
ys = shap_df_sum.abs_shap_values;
xs = shap_df_sum.feature_tag;
Makie.barplot!(xs, ys, direction = :x, width = 0.5, color = :green);

df_graph2 = combine(groupby(shap_df_sum, [:feature_name, :feature_tag]),
:feature_value => (x -> normalize(x)) => :feature_value_normalized);
ax = Makie.Axis(f[1, 2], yticks = (shap_df_sum0.feature_tag, shap_df_sum0.revised_labels), title = "(b)", xlabel = "Shapely value", xlabelsize = 42, ylabelsize = 42, yticksize = 12, yticklabelsize = 26, xticklabelsize = 30, yticklabelsvisible = false, titlesize = 50, titlealign = :left);
ylims!(0.25, length(unique(shap_df_sum.feature_name)) + 0.75);
ys = shap_df_sum.shap_effect;
xs = shap_df_sum.feature_tag;
Jitterplot.jitterplot!(xs, ys, 0.4, color = df_graph2.feature_value_normalized, colormap = color_map, direction = :horizontal, markersize = 20);
Makie.Colorbar(f[1, 3], colormap = color_map1, ticksvisible = false, ticks = ([0, 0.5, 1.0], ["Low values", "", "High values"]), ticklabelrotation = pi/2, height = Relative(1/3), ticklabelsize = 30)
f

Makie.save("outputs/Fig. 4. Feature importance for the best ensemble model sets predicting H2O2 production rate.png", f, px_per_unit = 3.3);
