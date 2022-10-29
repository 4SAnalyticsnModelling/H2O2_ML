using CSV, DataFrames, CairoMakie, Statistics, StatsBase;
cd(@__DIR__);
df = DataFrame();
for (cvtag, cvname) in zip(1:2, ["3-fold", "5-fold"])
	df0 = DataFrame(CSV.File("nnet_model_architechture_4_" * cvname * "_CV_h2o2_production_rate_trained_model/model_training_records.csv"))
	df0[!, :cv_name] .= cvname;
	df0[!, :cv_tag] .= cvtag;
	append!(df, df0)
end;
show(first(df, 10), allcols = true);

df_sum = combine(groupby(df, :cv_name),
:r_squared_train => (x -> median(x)) => :median_r2_training,
:r_squared_valid => (x -> median(x)) => :median_r2_validation,
:r_squared_test => (x -> median(x)) => :median_r2_test,
:rmse_train => (x -> median(x)) => :median_rmse_training,
:rmse_valid => (x -> median(x)) => :median_rmse_validation,
:rmse_test => (x -> median(x)) => :median_rmse_test,
:r_squared_train => (x -> mode(x)) => :modal_r2_training,
:r_squared_valid => (x -> mode(x)) => :modal_r2_validation,
:r_squared_test => (x -> mode(x)) => :modal_r2_test,
:rmse_train => (x -> mode(x)) => :modal_rmse_training,
:rmse_valid => (x -> mode(x)) => :modal_rmse_validation,
:rmse_test => (x -> mode(x)) => :modal_rmse_test
);
show(first(df_sum, 10), allcols = true);
rename!(df_sum, "cv_name" => "Cross-validation strategy");
CSV.write("outputs/Table to help describe Figure 2 cross-validation.csv", df_sum);



#  Graphics
f = Makie.Figure(resolution = (1400, 1100));
ax = Makie.Axis(f[1, 1], xticklabelsvisible = false, yticks = 0:0.1:1.0, titlesize = 28, titlealign = :left, ylabel = "Predicted vs. reported R²", ylabelsize = 24, title = "(a) Training", yticklabelsize = 20)
ylims!(ax, 0, 1.0);
for i in 1:length(unique(df.cv_tag))
	df_hist = df[df.cv_tag .== i, :];
	hist!(ax, df_hist.r_squared_train, color = (:green, 1.0), bins = 100, normalization = :probability, scale_to=-0.6, offset=i, direction=:x)
end
Makie.boxplot!(df.cv_tag, df.r_squared_train, show_notch = false, color = (:white, 0.0), strokewidth = 1);

ax = Makie.Axis(f[2, 1], xticklabelsvisible = true, yticks = 0:1.0:2000.0, titlesize = 28, titlealign = :left, ylabel = "Predicted vs. reported RMSE\n(kg m⁻³ d⁻¹)", ylabelsize = 24, xticks = (1.0:1.0:2.0, ["3-Fold", "5-Fold"]), xticklabelsize = 20, xticklabelrotation = pi/2, yticklabelsize = 20)
ylims!(ax, 0, 10.0);
for i in 1:length(unique(df.cv_tag))
	df_hist = df[df.cv_tag .== i, :];
	hist!(ax, df_hist.rmse_train, color = (:green, 1.0), bins = 50, normalization = :probability, scale_to=-0.6, offset=i, direction=:x)
end
Makie.boxplot!(df.cv_tag, df.rmse_train, show_notch = false, color = (:white, 0.0), strokewidth = 1);
ax = Makie.Axis(f[1, 2], xticklabelsvisible = false, yticks = 0:0.1:1.0, titlesize = 28, titlealign = :left, ylabel = "Predicted vs. reported R²", ylabelsize = 24, title = "(b) Validation", ylabelvisible = false, yticklabelsvisible = false)
ylims!(ax, 0, 1.0);
for i in 1:length(unique(df.cv_tag))
	df_hist = df[df.cv_tag .== i, :];
	hist!(ax, df_hist.r_squared_valid, color = (:green, 1.0), bins = 100, normalization = :probability, scale_to=-0.6, offset=i, direction=:x)
end
Makie.boxplot!(df.cv_tag, df.r_squared_valid, show_notch = false, color = (:white, 0.0), strokewidth = 1);
ax = Makie.Axis(f[2, 2], xticklabelsvisible = true, yticks = 0:1.0:2000.0, titlesize = 28, titlealign = :left, ylabel = "Predicted vs. reported RMSE\n(kg m⁻³ d⁻¹)", ylabelsize = 24, xticks = (1.0:1.0:2.0, ["3-Fold", "5-Fold"]), xticklabelsize = 20, ylabelvisible = false, yticklabelsvisible = false, xticklabelrotation=pi/2)
ylims!(ax, 0, 10.0);
for i in 1:length(unique(df.cv_tag))
	df_hist = df[df.cv_tag .== i, :];
	hist!(ax, df_hist.rmse_valid, color = (:green, 1.0), bins = 600, normalization = :probability, scale_to=-0.6, offset=i, direction=:x)
end
Makie.boxplot!(df.cv_tag, df.rmse_valid, show_notch = false, color = (:white, 0.0), strokewidth = 1);

ax = Makie.Axis(f[1, 3], xticklabelsvisible = false, yticks = 0:0.1:1.0, titlesize = 28, titlealign = :left, ylabel = "Predicted vs. reported R²", ylabelsize = 24, title = "(c) Test", ylabelvisible = false, yticklabelsvisible = false)
ylims!(ax, 0, 1.0);
for i in 1:length(unique(df.cv_tag))
	df_hist = df[df.cv_tag .== i, :];
	hist!(ax, df_hist.r_squared_test, color = (:green, 1.0), bins = 100, normalization = :probability, scale_to=-0.6, offset=i, direction=:x)
end
Makie.boxplot!(df.cv_tag, df.r_squared_test, show_notch = false, color = (:white, 0.0), strokewidth = 1);

ax = Makie.Axis(f[2, 3], xticklabelsvisible = true, yticks = 0:1.0:2000.0, titlesize = 28, titlealign = :left, ylabel = "Predicted vs. reported RMSE\n(kg m⁻³ d⁻¹)", ylabelsize = 24, xticks = (1.0:1.0:2.0, ["3-Fold", "5-Fold"]), xticklabelsize = 20, ylabelvisible = false, yticklabelsvisible = false, xticklabelrotation=pi/2)
ylims!(ax, 0, 10.0);
for i in 1:length(unique(df.cv_tag))
	df_hist = df[df.cv_tag .== i, :];
	hist!(ax, df_hist.rmse_test, color = (:green, 1.0), bins = 600, normalization = :probability, scale_to=-0.6, offset=i, direction=:x)
end
Makie.boxplot!(df.cv_tag, df.rmse_test, show_notch = false, color = (:white, 0.0), strokewidth = 1);
Makie.Label(f[3, :], "Cross validation strategies", textsize = 25);
f
Makie.save("outputs/Fig. 2. Cross validation model performance.png", px_per_unit = 3.3, f);
